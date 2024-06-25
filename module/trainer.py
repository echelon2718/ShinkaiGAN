import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

class Trainer:
    def __init__(self, 
                 generator, 
                 discriminator, 
                 gen_loss,
                 disc_loss,
                 batch_size,
                 learning_rate_G=0.0001, 
                 learning_rate_D=0.0001, 
                 max_depth=15,
                 display_step=20,
                 device="cuda"):
        
        self.gen = generator
        self.disc = discriminator
        self.loss_g = gen_loss
        self.loss_d = disc_loss
        self.batch_size = batch_size
        self.max_depth = max_depth
        self.display_step = display_step
        self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=learning_rate_G, betas=(0.5, 0.999))
        self.disc_opt = torch.optim.Adam(self.disc.parameters(), lr=learning_rate_D, betas=(0.5, 0.999))
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.gen_opt, step_size=1000, gamma=0.99)
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.disc_opt, step_size=1000, gamma=0.99)
        self.criterion = nn.BCEWithLogitsLoss()
        self.gen_losses = []
        self.disc_losses = []

    def train_progressive_gan(self, epochs, depth, train_loader):
        mean_generator_loss = 0
        mean_discriminator_loss = 0
        cur_step = 0
        
        for epoch in range(epochs):
            for real_A, real_B in tqdm(train_loader):
                cur_batch_size = len(real_A)
                real_A = real_A.to(device)
                real_B = real_B.to(device)

                # Update Discriminator #
                self.disc_opt.zero_grad()
                fake_B = self.gen(real_A, depth).detach()
                disc_loss_value = self.loss_d(real_B, fake_B, depth, self.disc)
                disc_loss_value.backward()
                # >> Gradient Clipping for Discriminator (to avoid exploding gradient)
                torch.nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=1.0)
                self.disc_opt.step()  # Update optimizer
                self.disc_losses.append(disc_loss_value.item())

                # Update Generator #
                self.gen_opt.zero_grad()
                gen_loss_value, fake_B = self.loss_g(real_A, real_B, depth, self.gen, self.disc)
                gen_loss_value.backward()
                # >> Gradient Clipping for Generator
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=1.0) 
                self.gen_opt.step()  # Update optimizer
                self.gen_losses.append(gen_loss_value.item())

                mean_discriminator_loss += disc_loss_value.item() / self.display_step
                mean_generator_loss += gen_loss_value.item() / self.display_step

                 ### Visualization and Logging ###
                if cur_step % self.display_step == 0 and cur_step > 0:
                    print(f"Epoch {epoch}/{epochs}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                    
                    mean_generator_loss = 0
                    mean_discriminator_loss = 0

                # Update learning rate schedulers
                self.scheduler_G.step()
                self.scheduler_D.step()
                
                cur_step += 1
            
    def progressive_training(self, epochs : list, levels : list, src_dir: str, tgt_dir: str):
        # levels = [2, 5, 8, 15]
        for i, depth in enumerate(levels):
            print(f"Starting training at depth {depth}")
            transform = transforms.Compose([
                transforms.CenterCrop((854, 854)),
                transforms.Resize(64 * 2**i, interpolation=Image.BILINEAR),
                transforms.ToTensor(),
            ])
            print("Initialize/Reinitialize Scheduler for every depth...")
            self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.gen_opt, step_size=1000, gamma=0.99)
            self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.disc_opt, step_size=1000, gamma=0.99)
            print("Success. Loading Dataset...")
            train_dataset = MakotoShinkaiDataset(src_dir, tgt_dir, transform, split="train")
            test_dataset = MakotoShinkaiDataset(src_dir, tgt_dir, transform, split="test")
            print("Dataset loaded! Initiating Data Loader...")
            train_loader = DataLoader(train_dataset, batch_size=16 if i == 0 else 8 if i == 1 else 4 if i == 2 else 2, shuffle=True, num_workers=15)
            test_loader = DataLoader(test_dataset, batch_size=16 if i == 0 else 8 if i == 1 else 4 if i == 2 else 2, shuffle=True, num_workers=15)
            print("Success. Starting progressive training.")
            
            gen.up_levels[depth].upsample = False
            self.train_progressive_gan(epochs[i], depth, train_loader)
            gen.up_levels[depth].upsample = True
            
            print(f'Completed training at depth {depth}')

    def reset_train_history(self):
        self.gen_losses = []
        self.disc_losses = []