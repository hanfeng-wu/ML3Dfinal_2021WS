import torch
import numpy as np
from time import time

class Trainer:
    def __init__(self,model, model_save_path,loss_function,  optimizer, batch_size,device,training_dataset, validation_dataset = None, score_function = None):
        self.model = model
        self.loss_function = loss_function
        self.score_function = score_function
        self.optimizer = optimizer
        self.device = device
        self.model_save_path = model_save_path
        self.training_dataloader = torch.utils.data.DataLoader(
            training_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
            batch_size=batch_size,   # The size of batches is defined here
            shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
            num_workers=0,   # Data is usually loaded in parallel by num_workers
            pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
        )
        if(validation_dataset is not None):
            self.validation_dataloader = torch.utils.data.DataLoader(
                validation_dataset,  
                batch_size=batch_size,  
                shuffle=True,   
                num_workers=0,   
                pin_memory=True  
                )
        else:
            self.validation_dataloader = None

        print("Total number of training batches:", len(self.training_dataloader))
    

    def move_batch_to_device(self, batch,):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch[0] = batch[0].to(self.device)
        batch[1] = batch[1].to(self.device)

    def fit(self,epochs=10,learning_rate=0.01):
        print("training on",self.device)
        torch.cuda.empty_cache()
        self.model.to(self.device)
        self.optimizer = self.optimizer(self.model.parameters(),learning_rate)
        best_loss = float('inf')

        for epoch in range(epochs):
            print("Epoch",epoch,":")
            epoch_start_time = time()
            avg_train_loss, avg_train_score, avg_time_per_training_batch = self.train_loop()
            epoch_end_time = time()
            total_epoch_time = epoch_end_time - epoch_start_time
            
            if(self.validation_dataloader is not None):
                val_start_time = time()
                avg_val_loss, avg_val_score, avg_time_per_validation_batch = self.eval_loop()
                val_end_time = time()
                total_validation_time = val_end_time - val_start_time

                if(avg_val_loss < best_loss):
                    best_loss = avg_val_loss
                    self.model.save(self.model_save_path)
            else:
                avg_val_loss = avg_val_score = avg_time_per_validation_batch = total_validation_time ="NA"
                if(avg_train_loss < best_loss):
                    best_loss = avg_train_loss
                    self.model.save(self.model_save_path)
            
            report = f"""
            avg time per training batch: {avg_time_per_training_batch} seconds 
            avg time per validation batch: {avg_time_per_validation_batch} seconds 
            time per epoch:     {total_epoch_time} seconds 
            time per validation {total_validation_time} seconds
            avg training loss:  {avg_train_loss}
            avg training score: {avg_train_score}
            avg validation loss:  {avg_val_loss}
            avg validation score: {avg_val_score}
            
            """
            print(report)



    def train_loop(self,):
        self.model.train()
        self.model.to(self.device)
        total_num_batches = len(self.training_dataloader)
        previously_printed = 0
        times = []
        losses = []
        scores = []
        start_batch_time = time()
        for batch_idx, batch in enumerate(self.training_dataloader):
            ######## VISUALIZATION ########
            total_percentage = int(((batch_idx+1)/total_num_batches) * 50)
            percentage_to_print = total_percentage - previously_printed
            previously_printed = total_percentage
            if(percentage_to_print<0):
                percentage_to_print = 0
            for _ in range(percentage_to_print):
                print("#",end='')
            ###############################

            self.optimizer.zero_grad()

            self.move_batch_to_device(batch)
            inputs, targets = batch
            
            preds = self.model(inputs)

            loss = self.loss_function(preds, targets)
            if(self.score_function is not None):
                score = self.score_function(preds,targets)
            else: 
                score = 0
            loss.backward()

            self.optimizer.step()

            losses.append(loss.item())
            scores.append(score)

            ############## TIMING ###############################
            end_batch_time = time()
            time_per_batch = end_batch_time - start_batch_time
            start_batch_time = end_batch_time
            times.append(time_per_batch)
            ######################################################
        avg_time_per_batch = np.mean(times)
        avg_train_loss = np.mean(losses)
        if(self.score_function is not None):
            avg_train_score = np.mean(scores)
        else:
            avg_train_score = "NA"
        print()
        return avg_train_loss, avg_train_score, avg_time_per_batch

    def eval_loop(self):
        self.model.eval()
        self.model.to(self.device)
        total_num_batches = len(self.validation_dataloader)
        previously_printed = 0
        times = []
        losses = []
        scores = []
        start_batch_time = time()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.validation_dataloader):
                ######## VISUALIZATION ########
                total_percentage = int(((batch_idx+1)/total_num_batches) * 50)
                percentage_to_print = total_percentage - previously_printed
                previously_printed = total_percentage
                if(percentage_to_print<0):
                    percentage_to_print = 0
                for _ in range(percentage_to_print):
                    print("#",end='')
                ###############################
                self.move_batch_to_device(batch)
                inputs, targets = batch
                
                preds = self.model(inputs)

                loss = self.loss_function(preds, targets)
                if(self.score_function is not None):
                    score = self.score_function(preds,targets)
                else: 
                    score = 0

                losses.append(loss.item())
                scores.append(score)

                ############## TIMING ###############################
                end_batch_time = time()
                time_per_batch = end_batch_time - start_batch_time
                start_batch_time = end_batch_time
                times.append(time_per_batch)
                ######################################################
            avg_time_per_batch = np.mean(times)
            avg_train_loss = np.mean(losses)
            if(self.score_function is not None):
                avg_train_score = np.mean(scores)
            else:
                avg_train_score = "NA"
        print()
        return avg_train_loss, avg_train_score, avg_time_per_batch