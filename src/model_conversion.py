import torch
import torchvision
from ball_tracker_net import BallTrackerNet
# An instance of your model.
# model = torch.load('/Users/tyler/Documents/GitHub/TennisProject/src/saved states/tracknet_weights_2_classes.pth')
# print(model)
# Switch the model to eval model
# model.eval()

detector = BallTrackerNet(out_channels=2)
saved_state_dict = torch.load('/Users/tyler/Documents/GitHub/TennisProject/src/saved states/tracknet_weights_2_classes.pth')
detector.load_state_dict(saved_state_dict['model_state'])
detector.eval()
# model = torch.load('/Users/tyler/Documents/GitHub/TennisProject/src/saved states/tracknet_weights_2_classes.pth')
# try:
#     model.eval()
# except AttributeError as error:
#     print(error)
### 'dict' object has no attribute 'eval'

# model.load_state_dict(model['state_dict'])
### now you can evaluate it
# model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
model_scripted = torch.jit.script(detector)

# Save the TorchScript model
model_scripted.save("traced_tracknet_model.pt")