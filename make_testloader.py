# to test the FNO you need to create a test loader

reader = MatReader('/content/drive/MyDrive/precipitation/precipitation.mat')
sub = 1
S = 256   
T_in = 52  
T = 12     
step = 1
batch_size = 2
# use the last 30 points for testing
test_a = reader.read_field('u')[-30:,::sub,::sub,:T_in]
test_u = reader.read_field('u')[-30:,::sub,::sub,T_in:T+T_in]
ntest = 30
test_a = test_a.reshape(ntest,S,S,T_in)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)
