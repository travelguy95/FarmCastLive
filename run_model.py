# now you can load the pretrained model and run the test
from matplotlib import pyplot as plt

modes, width = 21,36
model = FNO2d(modes, modes, width).cuda()
model.load_state_dict(torch.load('/content/drive/MyDrive/precipitation/21m_36w_1.pth'))
model = model.to('cuda')

batch_size = 2
with torch.no_grad():
  for xx, yy in test_loader:
    xx = xx.to(device)
    yy = yy.to(device)
    for t in range(0, T, step):
      y = yy[..., t:t + step]
      im = model(xx)
      xx = torch.cat((xx[..., step:], im), dim=-1)
      predicted = im.reshape(batch_size, -1)
      plt.imshow(np.rot90(predicted[0,:].reshape(256,256).cpu(),3))
      plt.show()
      actual = y.reshape(batch_size, -1)
      plt.imshow(np.rot90(actual[0,:].reshape(256,256).cpu(),3))
      plt.show()
      plt.imshow(np.rot90(predicted[1,:].reshape(256,256).cpu(),3))
      plt.show()
      actual = y.reshape(batch_size, -1)
      plt.imshow(np.rot90(actual[1,:].reshape(256,256).cpu(),3))
      plt.show()
