# Petal_FL

py -m grpc_tools.protoc -I=fl_app/proto --python_out=fl_app fl_app/proto/fl.proto

py -m fl_app.server_app.server
py -m fl_app.client_app.client -c 1 