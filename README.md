# Petal_FL

## For simple grpc
py -m grpc_tools.protoc -I=fl/proto --python_out=. --grpc_python_out=. fl/proto/simple_fl.proto

## For amble grpc
py -m grpc_tools.protoc -I=fl/proto --python_out=. --grpc_python_out=. fl/proto/amble_fl.proto

should I inclue GPU acceleration? Might reduce start time of PyTorch

Need to implement server refresh mechanism