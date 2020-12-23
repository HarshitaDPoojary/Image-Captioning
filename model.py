import torch
import torch.nn as nn
import torchvision.models as models



#Tried DesnseNet gave size mismatch error
# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
        
#         net = models.densenet161(pretrained = True)
#         for param in net.parameters():
#             param.requires_grad_(False)
#         self.net = net.features
#         self.embed = nn.Linear(net.classifier.in_features, embed_size)
#     def forward(self, images):
#         features = self.net(images)
#         features = self.embed(features)
#         return features

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
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
        super(DecoderRNN, self).__init__()
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.num_layers=num_layers
        
        # turning words into vector
        self.embed_caption=nn.Embedding(vocab_size,embed_size)
         # intializing LSTM
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        # mapping the dimension to vocab_size
        self.linear_layer_out=nn.Linear(hidden_size,vocab_size)
           
    
    def forward(self, features, captions):
        embeded=self.embed_caption(captions[:,:-1])
        
        input_embed=torch.cat((features.unsqueeze(dim=1),embeded),dim=1)
        
        lstm_output,_ =self.lstm(input_embed)
        #pass the lstm output to linear layer
        output=self.linear_layer_out(lstm_output)
        # now return the output
        return output
    
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        prediction=[]
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            
            outputs = self.linear_layer_out(lstm_out.squeeze(1))
            _, output = outputs.max(dim=1)                   
           
            # appending on list
            prediction.append(output.item())
            
            # for next iteration
            inputs = self.embed_caption(output)             
            inputs = inputs.unsqueeze(1)   
        return prediction

#         
# class DecoderRNN(nn.Module):
#     "Referring to https://www.deeplearningbook.org/contents/rnn.html usinf teacher forcer"
#     def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
#         super(DecoderRNN, self).__init__()
#         self.embed_size = embed_size
#         self.hidden_size = hidden_size
#         self.vocab_size = vocab_size 
#         self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
#         self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
#         self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
    
#     def forward(self, features, captions):
        
#         batch_size = features.size(0)
#         hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
#         cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
#         outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()
        
#         captions_embed = self.embed(captions)
#         for index in range(captions.size(1)):
#             if index == 0:
#                 hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
#             else:
#                 hidden_state, cell_state = self.lstm_cell(captions_embed[:, index, :], (hidden_state, cell_state))
            
#             out = self.fc_out(hidden_state)
            
#             outputs[:, index, :] = out
    
#         return outputs

#     def sample(self, inputs, states=None, max_len=20):
#         " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
#         batch_size = inputs.shape[0]
#         hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
#         cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
#         outputs = torch.empty((batch_size, max_len, self.vocab_size)).cuda()
#         output = []

        
#         for index in range(max_len):
#             print(inputs.size())
#             if index == 0:
#                 hidden_state, cell_state = self.lstm_cell(inputs[0], (hidden_state, cell_state))
#             else:
#                 hidden_state, cell_state = self.lstm_cell(inputs, (hidden_state, cell_state))
            
#             out = self.fc_out(hidden_state)
#             print(out)
#             _, predicted_index = torch.max(out, 1)
#             print(out.size())
#             output.append(predicted_index.cpu().numpy()[0].item()) 
            
#             if (predicted_index == 1 or len(output) >= max_len):
#                 # if reached the max length or predicted the end token
#                 break
            
            
#             inputs = self.embed(predicted_index)
#         return output
            
       