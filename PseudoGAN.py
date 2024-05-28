class PseudoGAN: 
    def __init__(self, n_channels, n_classes, latent_size, device): 
        self.latent_size = latent_size
        self.generator = Generator(latent_size, n_channels)
        self.classifier = Classifier(n_channels, n_classes)

        self.CEloss = nn.CrossEntropyLoss() 
      
        self.ice = ICE()
        self.device = device 
        self.to(device)
    
    def to(self, device): 
        self.generator.to(device) 
        self.classifier.to(device)
    
    def load_gen_state_dict(self, file):
        self.generator.load_state_dict(torch.load(file))
    
    @staticmethod
    def get_lr(optimizer: Type[optim.Optimizer]): 
        for param_group in optimizer.param_groups:
            return param_group['lr']
    @torch.no_grad()
    def accuracy(self, test_dl):
        self.classifier.eval()
        corrected = 0
        for b in tqdm(test_dl):
            images, y = b
            outs = self.classifier.forward(images)
            outs = torch.argmax(outs, dim=1)
            corrected += (outs == y).sum().item()
        return corrected / test_dl.num_data()
    
    
    def classifier_step(self, sup_imgs, sup_labels, unsup_imgs, batch_size): 
        # Loss for labeled samples
        sup_outs = self.classifier(sup_imgs) 
        sup_loss = self.CEloss(sup_outs, sup_labels)

        # Loss for unlabeled samples
        # Pseudo_label:  Pick up the class which
        # has maximum predicted probability for each unlabeled
        # sample
        unsup_outs = self.classifier(unsup_imgs) 
        unsup_pseudolabels = torch.argmax(unsup_outs, dim = 1) 
        # print(unsup_pseudolabels.shape)
        unsup_loss = self.CEloss(unsup_outs, unsup_pseudolabels)

        # Loss for generated samples. Also pseudo_labelling as for 
        # unsup imgs, but now apply the inverted binary cross entropy 
        # as loss. Aim: decrease the margin of these data points
        # and make the prediction distribution flat
        z = torch.randn([batch_size, self.latent_size, 1, 1], device = self.device)
        fake_imgs = self.generator(z) 
        fake_outs = self.classifier(fake_imgs)
        fake_pseudolabels = torch.argmax(fake_outs, dim = 1) 
        fake_loss = self.ice(fake_outs, fake_pseudolabels) 

        return sup_loss + (unsup_loss + fake_loss)/2

    def fit(self, epochs, batch_size, batch_per_epoch, max_lr, sup_ds:CustomDataSet, unsup_ds:CustomDataSet, test_dl, optim:Type[optim.Optimizer], weight_decay = 0, sched = True, PATH = ".", save = False, grad_clip = False): 
        history: dict[str, list] = {'epochs': epochs, 'Loss': []}
        if sched: 
            history['Learning rate'] = []
        optimizerC = optim(self.classifier.parameters(), lr = max_lr, weight_decay = weight_decay)

        if sched: 
            OneCycleLR = torch.optim.lr_scheduler.OneCycleLR(optimizerC, max_lr, epochs=epochs, steps_per_epoch=batch_per_epoch)
        
        with open('check.txt', 'w') as f: 
            for epoch in (range(epochs)):
                lrs = []
                self.classifier.train() 
                for i in (tqdm(range(batch_per_epoch))): 
                    sup_imgs, labels = random_split(sup_ds, [batch_size, len(sup_ds) - batch_size])[0][:]
                    unsup_imgs = random_split(unsup_ds, [batch_size, len(unsup_ds) - batch_size])[0][:]

                    # train classifier
                    C_loss = self.classifier_step(sup_imgs.to(self.device), labels.to(self.device), unsup_imgs.to(self.device), batch_size) 
                    C_loss.backward()
                    
                    optimizerC.step()
                    lrs.append(self.get_lr(optimizerC))
                    history['Loss'].append(C_loss.item())
                    # optimizerC.zero_grad()
                    
                    if grad_clip: 
                        nn.utils.clip_grad_value_(self.classifier.parameters(), 0.1)
                    if sched: 
                        OneCycleLR.step()
                    
                    tqdm.write(f'C_loss: {C_loss.detach().item()}', end = "\r")
                    
                self.classifier.eval()
                acc = self.accuracy(test_dl)
                f.write(f'accuracy: {acc}\n')
                tqdm.write(f'accuracy: {acc}', end = "\r")
                if sched: 
                    history['Learning rate'] += lrs
                
        if save: 
            torch.save(self.classifier.state_dict(), PATH + '.pt')
    
        return history