import torch
import torchvision
from ball_tracker_net import BallTrackerNet
import torch.onnx 
# An instance of your model.
# model = torch.load('/Users/tyler/Documents/GitHub/TennisProject/src/saved states/tracknet_weights_2_classes.pth')
# print(model)
# Switch the model to eval model
# # model.eval()

# detector = BallTrackerNet()
# saved_state_dict = torch.load('/Users/tyler/Documents/GitHub/TennisProject/src/saved states/tracknet_weights_2_classes.pth')
# detector.load_state_dict(saved_state_dict['model_state'])
# detector.eval()
# # model = torch.load('/Users/tyler/Documents/GitHub/TennisProject/src/saved states/tracknet_weights_2_classes.pth')
# # try:
# #     model.eval()
# # except AttributeError as error:
# #     print(error)
# ### 'dict' object has no attribute 'eval'

# # model.load_state_dict(model['state_dict'])
# ### now you can evaluate it
# # model.eval()

# # An example input you would normally provide to your model's forward() method.
# example = torch.rand(1, 3, 224, 224)

# # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# model_scripted = torch.jit.script(detector)

# # Save the TorchScript model
# model_scripted.save("traced_tracknet_model.pt")



#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    detector.eval() 

    # Let's create a dummy input tensor  
    # dummy_input = torch.randn(1, 2, requires_grad=True)  
    dummy_input = torch.randn(1, 2, 224, 224, requires_grad=True)  

    # Export the model   
    torch.onnx.export(detector,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "ImageClassifier.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 

if __name__ == "__main__": 

    # Let's build our model 
    #train(5) 
    #print('Finished Training') 

    # Test which classes performed well 
    #testAccuracy() 

    # Let's load the model we just created and test the accuracy per label 
    # model = Network() 
    # path = "myFirstModel.pth" 
    # model.load_state_dict(torch.load(path)) 
    detector = BallTrackerNet(out_channels=2)
    saved_state_dict = torch.load('/Users/tyler/Documents/GitHub/TennisProject/src/saved states/tracknet_weights_2_classes.pth')
    detector.load_state_dict(saved_state_dict['model_state'])
    # detector.eval()

    # Test with batch of images 
    #testBatch() 
    # Test how the classes performed 
    #testClassess() 

    # Conversion to ONNX 
    Convert_ONNX() 