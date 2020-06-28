import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super().__init__()
        
        self.embd = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM( input_size = embed_size, 
                             hidden_size = hidden_size, 
                             num_layers = num_layers, 
                             dropout = 0, 
                             batch_first=True
                           )
        self.fc   = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        captions  = captions[:,:-1]
        captions  = self.embd(captions)
        features  = features.unsqueeze(1)
        inputs    = torch.cat((features,captions), dim = 1)
        outputs, _ = self.lstm(inputs)
        outputs    = self.fc(outputs)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        outputs = []
        
        while(len(outputs)<max_len+1):
            
            output , states = self.lstm(inputs,states)
            output          = output.squeeze(dim = 1)
            output          = self.fc(output)
            
            _ , output = torch.max(output,1)
            
            outputs.append(output.cpu().numpy()[0].item())
            
            if output == 1 :
                break
            
            inputs = self.embd(output)
            
            inputs = inputs.unsqueeze(1)
            
        return outputs
            
        