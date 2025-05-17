# Petal_FL

py -m grpc_tools.protoc -I=fl_app/proto --python_out=fl_app --grpc_python_out=fl_app fl_app/proto/fl.proto
- improperly imports fl_pb2 to fl_pb2_grpc, need to fix

py -m fl_app.server_app.server
py -m fl_app.client_app.client -c 1 

Model architecture should be shared ahead of time. Likely at connect msg.

should I inclue GPU acceleration? Might reduce start time of PyTorch

Need to implement server refresh mechanism