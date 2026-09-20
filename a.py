import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
print(device)
a = torch.zeros(2, 3, device=device)
print(a)