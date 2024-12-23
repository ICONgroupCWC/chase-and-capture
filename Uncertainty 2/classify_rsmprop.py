from pymongo import MongoClient
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def retrieveRandomFromMongo(num_entries, agentID, lossFn, query=None):
    client = MongoClient("mongodb://localhost:27017/")
    db = client['sequentialdata']
    collection = db[f'mode_30_Q_Agent{agentID}'] 
    
    if query is None:
        query = {}

    all_documents = list(collection.find(query))
    if num_entries >= len(all_documents):
        # If num_entries is greater than or equal to the total number of documents, return all documents
        retDocs =  all_documents

    else:
        # Randomly select the required number of entries
        sampled_documents = random.sample(all_documents, num_entries)
        retDocs =  sampled_documents

    outputSamples = []
    if lossFn == "MSE":
        for doc in retDocs:
            x = np.array(doc["obs"], dtype=np.float32)
            prev_actions = np.array(doc["prev_qvals"], dtype=np.float32)
            norm_prev_act = (prev_actions - min(prev_actions)) / sum(prev_actions - min(prev_actions))
            outputSamples.append((x, norm_prev_act))

    return outputSamples




class Network(nn.Module):
  def __init__(self):
    super(Network, self).__init__()
    self.linear1 = nn.Linear(input_dim, hidden_layer_1)
    self.bn1 = nn.BatchNorm1d(hidden_layer_1)

    self.linear2 = nn.Linear(hidden_layer_1, hidden_layer_2)
    self.bn2 = nn.BatchNorm1d(hidden_layer_2)


    self.linear3 = nn.Linear(hidden_layer_2, hidden_layer_3)
    self.bn3 = nn.BatchNorm1d(hidden_layer_3)


    self.linear4 = nn.Linear(hidden_layer_3, output_dim)
    self.relu = nn.ReLU()


  def forward(self, x):
    x = self.linear1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.linear3(x)
    x = self.bn3(x)
    x = self.relu(x)
    x = self.linear4(x)
    return x
  


def init_plot():
    plt.ion()  # Enable interactive mode
    plt.figure(figsize=(12, 5))
    
    # Create empty plots for loss and accuracy
    loss_line, = plt.plot([], [], label="Loss", color='b')
    accuracy_line, = plt.plot([], [], label="Accuracy", color='g')
    
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Training Loss and Accuracy over Epochs')
    # plt.xlim(0, EPOCHS)  # Set x-axis limit
    plt.ylim(0, 1)       # Set y-axis limit for accuracy (0 to 100 for percentage)
    plt.legend()
    plt.grid()
    
    return loss_line, accuracy_line

def update_plot(loss_line, accuracy_line, epoch, loss, accuracy):
    # Update the data in the plots
    loss_line.set_xdata(np.arange(1, epoch + 1))
    loss_line.set_ydata(losses[:epoch])  # Assuming `losses` is a list of losses
    accuracy_line.set_xdata(np.arange(1, epoch + 1))
    accuracy_line.set_ydata(accuracies[:epoch])  # Assuming `accuracies` is a list of accuracies
    
    plt.xlim(0, EPOCHS)  # Keep x-axis limit
    plt.ylim(0, 100)     # Keep y-axis limit for accuracy

    plt.draw()          # Redraw the plot
    plt.pause(0.01)     # Pause to allow the plot to update

if __name__ == "__main__":
    import wandb
    wandb.init(project="Mode 30",name=f"classify",save_code=True)

    input_dim = 10
    hidden_layer_1 = 200
    hidden_layer_2 = 400
    hidden_layer_3 = 200
    output_dim = 5
    BATCH_SIZE  = 2000
    EPOCHS = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochName = 4990
    INPUT_QNET_NAME = f'artifacts/models/{epochName}_mode30_agent_1.pt'
    FROM_SCRATCH = False



    clf = Network().to(device)  # Move the model to the appropriate device
    tgt = Network().to(device)  # Move the model to the appropriate device

    if not FROM_SCRATCH:
        clf.load_state_dict(torch.load(INPUT_QNET_NAME, map_location=torch.device('cpu')))


    optimizer = optim.Adam(clf.parameters(), lr=1e-4)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)
    

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
    #                  base_lr = 1e-4, # Initial learning rate which is the lower boundary in the cycle for each parameter group
    #                  max_lr = 1e-2, # Upper learning rate boundaries in the cycle for each parameter group
    #                  step_size_up = 1000, # Number of training iterations in the increasing half of a cycle
    #                  mode = "triangular",
    #                  cycle_momentum=False)
    
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
    #                     milestones=[200,400,600,800,1000,1200,1400,1600,1800,2000], # List of epoch indices
    #                     gamma =10) # Multiplicative factor of learning rate decay
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                   step_size = 500, # Period of learning rate decay
                   gamma = 0.5) # Multiplicative factor of learning rate decay
    

    

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
    #                                     T_0 = 500,# Number of iterations for the first restart
    #                                     T_mult = 2, # A factor increases TiTi​ after a restart
    #                                     eta_min = 1e-8) # Minimum learning rate
    

    criterion = nn.MSELoss()

    samples = retrieveRandomFromMongo(50000, 1, "MSE")
    data_loader = torch.utils.data.DataLoader(samples,
                                              batch_size=BATCH_SIZE,
                                              shuffle=True)

    # Lists to store loss and accuracy over epochs
    losses = []
    accuracies = []

    # Initialize the plot
    loss_line, accuracy_line = init_plot()
    prev_gradients = None

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        total_norm = 0
        total_norm_after = 0
        running_loss = .0
        n_batches = 0
        pred_perc = []

        innerProd = []

        for data in data_loader:
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = clf(inputs)

            loss = criterion(outputs, labels)

            loss.backward()


            current_gradients = [param.grad.clone() if param.grad is not None else None for param in clf.parameters()]

            if prev_gradients is not None and current_gradients is not None:
                inner_product = 0.0
                for prev, current in zip(prev_gradients, current_gradients):
                    if prev is not None and current is not None:
                        inner_product += torch.sum(prev * current).item()
                        innerProd.append(inner_product)

            prev_gradients = current_gradients



            
            
            for param in clf.parameters():
                if param.grad is not None:
                    # Compute the norm of the gradient for this parameter
                    param_norm = param.grad.data.norm(2)  # L2 norm
                    total_norm += param_norm.item() ** 2  # Accumulate norm squared


            
            torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=10.0)

            for param in clf.parameters():
                if param.grad is not None:
                    # Compute the norm of the gradient for this parameter
                    param_norm = param.grad.data.norm(2)  # L2 norm
                    total_norm_after += param_norm.item() ** 2  # Accumulate norm squared


            optimizer.step()


            correct_pred_percentage = (outputs.max(-1).indices == labels.max(-1).indices).sum().item() * 100 / BATCH_SIZE
            pred_perc.append(correct_pred_percentage)

            running_loss += loss.item()
            n_batches += 1


        total_norm = total_norm ** 0.5  # Final L2 norm of the gradients
        print(f'Epoch: {epoch}, Batch Gradient Norm: {total_norm}')

        total_norm_after = total_norm_after ** 0.5  # Final L2 norm of the gradients
        print(f'Epoch: {epoch}, Batch Gradient Norm after: {total_norm_after}')



        # Average loss and prediction accuracy for the epoch
        epochAvg_predPerc = np.mean(pred_perc)
        epochAvg_loss = running_loss / n_batches

        # Store metrics for plotting
        losses.append(epochAvg_loss)
        accuracies.append(epochAvg_predPerc)

        # Print metrics
        print(f"Epoch {epoch + 1}/{EPOCHS}, Prediction Accuracy = {epochAvg_predPerc}%, Loss = {epochAvg_loss}")

        wandb.log({'loss':running_loss / n_batches,
                   'Prediction Accuracy' : epochAvg_predPerc,
                   'Norm before clipping' : total_norm,
                   'Norm after clipping' : total_norm_after,
                   'Learning Rate': optimizer.param_groups[0]['lr'],
                   'AvgInnerProd' : np.mean(innerProd),
                   },step=epoch) 

        # Update the plot with the latest data
        # update_plot(loss_line, accuracy_line, epoch + 1, epochAvg_loss, epochAvg_predPerc)
        if epoch % 10 ==0:
            torch.save(clf.state_dict(), f'artifacts/models/trial3/{epoch}_mode30_agent_1.pt')

        scheduler.step()

    plt.ioff()  # Turn off interactive mode when done
    plt.show()  # Show the final plot