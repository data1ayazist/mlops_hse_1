# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import warnings

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
import model_service_pb2 as model__service__pb2

GRPC_GENERATED_VERSION = '1.67.1'
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower
    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f'The grpc package installed is at version {GRPC_VERSION},'
        + f' but the generated code in model_service_pb2_grpc.py depends on'
        + f' grpcio>={GRPC_GENERATED_VERSION}.'
        + f' Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}'
        + f' or downgrade your generated code using grpcio-tools<={GRPC_VERSION}.'
    )


class ModelServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.TrainModel = channel.unary_unary(
                '/model_service.ModelService/TrainModel',
                request_serializer=model__service__pb2.TrainModelRequest.SerializeToString,
                response_deserializer=model__service__pb2.TrainModelResponse.FromString,
                _registered_method=True)
        self.RetrainModel = channel.unary_unary(
                '/model_service.ModelService/RetrainModel',
                request_serializer=model__service__pb2.RetrainModelRequest.SerializeToString,
                response_deserializer=model__service__pb2.RetrainModelResponse.FromString,
                _registered_method=True)
        self.Predict = channel.unary_unary(
                '/model_service.ModelService/Predict',
                request_serializer=model__service__pb2.PredictRequest.SerializeToString,
                response_deserializer=model__service__pb2.PredictResponse.FromString,
                _registered_method=True)
        self.DeleteModel = channel.unary_unary(
                '/model_service.ModelService/DeleteModel',
                request_serializer=model__service__pb2.DeleteModelRequest.SerializeToString,
                response_deserializer=model__service__pb2.DeleteModelResponse.FromString,
                _registered_method=True)
        self.GetModelTypes = channel.unary_unary(
                '/model_service.ModelService/GetModelTypes',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=model__service__pb2.ModelTypesResponse.FromString,
                _registered_method=True)
        self.HealthCheck = channel.unary_unary(
                '/model_service.ModelService/HealthCheck',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                _registered_method=True)


class ModelServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def TrainModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RetrainModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Predict(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetModelTypes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def HealthCheck(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'TrainModel': grpc.unary_unary_rpc_method_handler(
                    servicer.TrainModel,
                    request_deserializer=model__service__pb2.TrainModelRequest.FromString,
                    response_serializer=model__service__pb2.TrainModelResponse.SerializeToString,
            ),
            'RetrainModel': grpc.unary_unary_rpc_method_handler(
                    servicer.RetrainModel,
                    request_deserializer=model__service__pb2.RetrainModelRequest.FromString,
                    response_serializer=model__service__pb2.RetrainModelResponse.SerializeToString,
            ),
            'Predict': grpc.unary_unary_rpc_method_handler(
                    servicer.Predict,
                    request_deserializer=model__service__pb2.PredictRequest.FromString,
                    response_serializer=model__service__pb2.PredictResponse.SerializeToString,
            ),
            'DeleteModel': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteModel,
                    request_deserializer=model__service__pb2.DeleteModelRequest.FromString,
                    response_serializer=model__service__pb2.DeleteModelResponse.SerializeToString,
            ),
            'GetModelTypes': grpc.unary_unary_rpc_method_handler(
                    servicer.GetModelTypes,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=model__service__pb2.ModelTypesResponse.SerializeToString,
            ),
            'HealthCheck': grpc.unary_unary_rpc_method_handler(
                    servicer.HealthCheck,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'model_service.ModelService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers('model_service.ModelService', rpc_method_handlers)


 # This class is part of an EXPERIMENTAL API.
class ModelService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def TrainModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/model_service.ModelService/TrainModel',
            model__service__pb2.TrainModelRequest.SerializeToString,
            model__service__pb2.TrainModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def RetrainModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/model_service.ModelService/RetrainModel',
            model__service__pb2.RetrainModelRequest.SerializeToString,
            model__service__pb2.RetrainModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def Predict(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/model_service.ModelService/Predict',
            model__service__pb2.PredictRequest.SerializeToString,
            model__service__pb2.PredictResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def DeleteModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/model_service.ModelService/DeleteModel',
            model__service__pb2.DeleteModelRequest.SerializeToString,
            model__service__pb2.DeleteModelResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def GetModelTypes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/model_service.ModelService/GetModelTypes',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            model__service__pb2.ModelTypesResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)

    @staticmethod
    def HealthCheck(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(
            request,
            target,
            '/model_service.ModelService/HealthCheck',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True)
