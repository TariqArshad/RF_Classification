from Processor import  POWDERRF_Processor
from Modeling import Conv1D_RF_Classifier
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
from safetensors.torch import load_file
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json
from time import perf_counter
import os

def Parse_Args():
    parser = argparse.ArgumentParser();
    parser.add_argument("--data_dir", type=str, default = "GlobecomPOWDER/")
    parser.add_argument("--task", type=str, default = "transmitter", choices = ["transmitter", "protocol", "joint"],help = "options are transmitter, protocol, or joint")
    parser.add_argument("--save_dir", type=str, default = "./")
    parser.add_argument("--model_checkpoint", type = str, default = "./")
    #parser.add_argument("--sample_len", type=int, default = 1024)
    parser.add_argument("--max_samples", type=int, default = 10000)
    parser.add_argument("--data_day", nargs = "+",type=int, default = [1, 2], help = "takes input as --data_day 1 2")
    parser.add_argument("--batch_sz", type=int, default = 128)
    parser.add_argument("--lr", type=float, default = 0.001)
    parser.add_argument("--epochs", type=int, default = 25)
    parser.add_argument("--checkpoint", type=int, default = 100)
    parser.add_argument("--signal_type", type = str, default = "All",choices = ["Mag", "Phase", "Complex", "Real", "Imag", "All"] , help = "options are Mag, Phase, Complex, Real, Imag, or All")
    parser.add_argument("--device", type = str, default = "cuda")
    parser.add_argument("--eval_only", action = 'store_true')

    args = parser.parse_args();
    return args;


def train(model = None, xtrain = np.array([]), ytrain = np.array([]), batch_sz = 100, epochs = 100, lr = 0.001 , savepath = "./", i_checkpoint = 25, device = "cuda", val_split = 0.05):
    """
    """

    if val_split != 0:
        val_index = int(np.floor(xtrain.shape[0]*val_split));
        xval = torch.tensor(xtrain[:val_index]).type(torch.float);
        yval = np.argmax(ytrain[:val_index], axis = 1);
        
        xtrain = xtrain[val_index:];
        ytrain = ytrain[val_index:];
        
        
    xtrain_batched = torch.tensor(xtrain).type(torch.float);
    xtrain_batched = torch.split(xtrain_batched, batch_sz);
    ytrain_batched = torch.argmax(torch.tensor(ytrain),dim = -1);
    ytrain_batched = torch.split(ytrain_batched, batch_sz);
    history = {"Avg_loss":[], "loss":[]}
    loss_list = [];
    loss_fn = torch.nn.CrossEntropyLoss();
    optimizer = torch.optim.Adam(model.parameters(),lr = lr);
    model = model.to(device)
    loss_sum = 0;
    i = 0;
    time = 0;
    history = {};
    for epoch in range(epochs):
        history[epoch] = {};
        print("Epoch:", epoch);
        t0 = perf_counter();
        for x_batch, y_batch in tqdm(zip(xtrain_batched, ytrain_batched)):
            i += 1;
            # Zero your gradients for every batch
            optimizer.zero_grad()
            x_batch = x_batch.to(device);
            y_batch = y_batch.flatten().to(device);
            #logits should be shape:(batch_sz, hidden_length)
            logits, loss = model(data_in = x_batch, y_true = y_batch);
            # Compute the loss and its gradients
            loss.backward();
            loss_sum += loss.item();
            avg_loss = loss_sum/i;
            if i%25 == 0:
                print("avg loss:", avg_loss)
            if i%i_checkpoint == 0:
                model_savepath = savepath + "checkpoint-" + str(i) + ".pth";
                torch.save(model.state_dict(), model_savepath);            
            # Adjust learning weights
            optimizer.step();

        t1 = perf_counter();
        time = time + (t1 - t0);
        history[epoch]["time"] = time;
        if val_split != 0:
            xval = xval.to(device)
            logits = model(xval)
            probs = torch.nn.functional.softmax(logits, dim = -1);
            preds = torch.argmax(probs, dim = -1).detach().cpu();
            acc_score = accuracy_score(preds.numpy(), yval);
            history[epoch]["val_acc"] = acc_score;
            print("Val acc:", acc_score)

    if savepath[-1] != "/": savepath += "/";
    model_savepath = savepath + "final_model"+ ".pth";
    torch.save(model.state_dict(), model_savepath); 

    history_savepath = savepath + "train_history.json";
    with open(history_savepath, "w") as f:
        json.dump(history, f);
    return model;
    
def evaluate(model = None, xtest = np.array([]), ytest = np.array([]), batch_sz = 32,savepath = "./" ,device = "cuda", labels = []):
    model = model.to(device)
    xtest_batched = torch.tensor(xtest).type(torch.float);
    xtest_batched = torch.split(xtest_batched, batch_sz);
    embeddings = [];
    for batch in tqdm(xtest_batched):
        batch = batch.to(device)
        embeddings.append(model(batch).detach().cpu().float().numpy())
    embeddings = np.concatenate(embeddings, axis = 0)
    preds = torch.nn.functional.softmax(torch.tensor(embeddings), dim = -1);
    preds = torch.argmax(preds, dim = -1).numpy();
    
    SNE_embedds = TSNE(n_components=2, learning_rate='auto', init = 'random').fit_transform(embeddings)

    ysamples = np.argmax(ytest, axis=1);
    
    ysamples_, preds_ = [], [];
    for ysample, predsample in zip(ysamples, preds):
        ysamples_.append(labels[ysample]);
        preds_.append(labels[predsample])
    
    fig, ax = plt.subplots(figsize=(13, 13)) 
    report = classification_report(ysamples_, preds_, output_dict = True, target_names = labels);
    cm = confusion_matrix(ysamples_, preds_,labels = labels);
    # Normalize confusion matrix to get percentages
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis];
    plt.rcParams.update({'font.size': 15})  # Set all font sizes to 16
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels).plot(ax = ax, colorbar = False, values_format=".2f");
    plt.xticks(rotation=90)
    cmplot_savepath = savepath + "confusion_matrix.png"
    plt.savefig(cmplot_savepath)

    plt.figure(figsize = (13, 13))
    plt.rcParams.update({'font.size': 16})  # Set all font sizes to 16
    yclasses = np.unique(ysamples);
    SNE_Embedds_dict = {};

    # Define a list of unique colors for plots
    colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", 
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#f0e442", "#d96f37"
    ]
    #colors = plt.cm.Set3(np.linspace(0, 1, len(yclasses)));
    for i in range(len(yclasses)):
        y_class = yclasses[i];
        color = colors[i];
        class_index = (ysamples == y_class)
        embedds = SNE_embedds[class_index];
        SNE_Embedds_dict[y_class] = embedds;
        plt.scatter(embedds[:, 0], embedds[:, 1], label = labels[y_class], color = color);
    plt.legend();
    SNEplot_savepath = savepath + "SNE_Embeddings.png";
    plt.savefig(SNEplot_savepath);

    results_path  = savepath + "results.json";
    print(report)
    with open(results_path, 'w') as f:
        json.dump(report, f);
    results = {};
    results["report"] = report;
    results["confusion matrix"] = cm;
    results["SNE embeddings"] = SNE_Embedds_dict

    return results  

def main(args):
    
    if args.task == "transmitter": classes = 4;
    elif args.task == "protocol": classes = 3;
    elif args.task == "joint": classes = 12;

    if os.path.isdir(args.save_dir) != True:
        os.mkdir(args.save_dir);

    print("Loading data...")
    processor = POWDERRF_Processor(datadir = args.data_dir, sample_len = 1024, samples_per_sig = 10, max_samples = args.max_samples, data_days = args.data_day, savedir = args.save_dir);
    xtrain, ytrain, xtest, ytest = processor(train_test_split=.8, signal_type=args.signal_type, task = args.task, loaddatadict = False);
    
    model = Conv1D_RF_Classifier(classes = classes)
    
    if args.eval_only == False:
        print("Beginnig Training...")
        model = train(model = model, xtrain = xtrain, ytrain = ytrain,  batch_sz = args.batch_sz, lr = args.lr ,epochs = args.epochs, savepath = args.save_dir, i_checkpoint = args.checkpoint, device = args.device)
        print("Training Finished!");
    print("Beginning Evaluation...")
    if args.eval_only:
        model_checkpoint = torch.load(args.model_checkpoint);
        model.load_state_dict(model_checkpoint);
        model.eval();
    results = evaluate(model = model, xtest = xtest, ytest = ytest, batch_sz = 32,savepath = args.save_dir ,device = args.device, labels = processor.label_encoder.categories_[0]);
    print("Evaluation Finished!")
    
    return 0;

if __name__ == "__main__":
    args = Parse_Args();
    main(args);

    