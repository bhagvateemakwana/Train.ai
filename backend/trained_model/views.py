from datetime import datetime
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from django.http import Http404, HttpResponse
from django.core.exceptions import ValidationError
from accounts.models import User

import requests
import json
import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import logging
import os

from .models import TrainedModel
from .serializer import TrainedModelSerializer, ModelStatsSerializer, ModelGraphSerializer
from ml_utils.pdf_generator import ModelReportGenerator

logger = logging.getLogger(__name__)

class UserTrainedModelView(APIView):
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request):
        try:
            userId = request.user.id
            trained_models = TrainedModel.objects.filter(user_id=userId)
            serializer = TrainedModelSerializer(trained_models, many=True)
            return Response({
                "message": "User trained models retrieved successfully.",
                "data": serializer.data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error retrieving user trained models for user {request.user.id}: {str(e)}")
            return Response({
                "error": "An error occurred while retrieving your trained models.",
                "message": "Please try again later."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ModelListView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        try:
            trained_models = TrainedModel.objects.filter(is_public=True)
            serializer = TrainedModelSerializer(trained_models, many=True)
            return Response({
                "message": "Public models retrieved successfully.",
                "data": serializer.data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Error retrieving public models: {str(e)}")
            return Response({
                "error": "An error occurred while retrieving public models.",
                "message": "Please try again later."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ModelDetailView(APIView):
    
    permission_classes = [IsAuthenticated]
    
    def get_object(self, pk):
        try:
            return TrainedModel.objects.get(pk=pk)
        except TrainedModel.DoesNotExist:
            raise Http404("Model not found.")
        except ValidationError:
            raise Http404("Invalid model ID format.")
        
    def get(self, request, pk):
        try:
            trained_model = self.get_object(pk)
            
            model_coefficients = None
            model_intercept = None
            
            if trained_model.model_file:
                try:
                    model_file_path = trained_model.model_file.path
                    
                    if not os.path.exists(model_file_path):
                        logger.warning(f"Model file not found: {model_file_path}")
                    else:
                        model = joblib.load(model_file_path)
                        model_coefficients = model.coef_.tolist() if hasattr(model, "coef_") else None
                        model_intercept = model.intercept_.tolist() if hasattr(model, "intercept_") else None
                        
                except (FileNotFoundError, EOFError, ValueError) as e:
                    logger.error(f"Error loading model file for model {pk}: {str(e)}")
                except Exception as e:
                    logger.error(f"Unexpected error loading model file for model {pk}: {str(e)}")
            
            return Response({
                "message": "Model data retrieved successfully.",
                "model": {
                    "id": str(trained_model.id),
                    "user_id": str(trained_model.user_id),
                    "name": trained_model.model_name,
                    "type": trained_model.model_type,
                    "polynomial_degree": trained_model.polynomial_degree,
                    "target_column": trained_model.target_column,
                    "features": trained_model.features,
                    "is_public": trained_model.is_public,
                    "likes": trained_model.likes,
                    "created_at": trained_model.created_at,
                    "model_file": trained_model.model_file.url if trained_model.model_file else None,
                },
                "metrics": ModelStatsSerializer(trained_model.stats).data if hasattr(trained_model, "stats") else {},
                "graphs": ModelGraphSerializer(trained_model.graphs.all(), many=True).data,
                "coefficients": model_coefficients,
                "intercept": model_intercept,
            }, status=status.HTTP_200_OK)
            
        except Http404 as e:
            return Response({
                "error": "Model not found.",
                "message": str(e)
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error retrieving model details for model {pk}: {str(e)}")
            return Response({
                "error": "An error occurred while retrieving model details.",
                "message": "Please try again later."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def post(self, request, pk):
        try:
            trained_model = self.get_object(pk)
            
            # Check if model file exists
            if not trained_model.model_file:
                return Response({
                    "error": "Model file not available.",
                    "message": "This model cannot be used for predictions."
                }, status=status.HTTP_400_BAD_REQUEST)
            
            model_file_path = trained_model.model_file.path
            
            # Check if file exists on disk
            if not os.path.exists(model_file_path):
                logger.error(f"Model file not found on disk: {model_file_path}")
                return Response({
                    "error": "Model file not found.",
                    "message": "The model file is missing from the server."
                }, status=status.HTTP_404_NOT_FOUND)

            try:
                model = joblib.load(model_file_path)
            except (FileNotFoundError, EOFError, ValueError) as e:
                logger.error(f"Error loading model file {model_file_path}: {str(e)}")
                return Response({
                    "error": "Failed to load model.",
                    "message": "The model file appears to be corrupted or invalid."
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Validate input features
            features_input = request.data.get('features')
            if not features_input:
                return Response({
                    "error": "Missing required field.",
                    "message": "'features' field is required."
                }, status=status.HTTP_400_BAD_REQUEST)
            
            try:
                features_array = np.array(features_input, dtype=float).reshape(1, -1)
            except (ValueError, TypeError) as e:
                return Response({
                    "error": "Invalid feature values.",
                    "message": "All feature values must be numeric."
                }, status=status.HTTP_400_BAD_REQUEST)

            # Make prediction
            try:
                prediction = model.predict(features_array)
                return Response({
                    "message": "Prediction completed successfully.",
                    "prediction": prediction.tolist()
                }, status=status.HTTP_200_OK)
            except Exception as e:
                logger.error(f"Prediction failed for model {pk}: {str(e)}")
                return Response({
                    "error": "Prediction failed.",
                    "message": "The model could not process the provided features. Please check the input format and values."
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Http404:
            return Response({
                "error": "Model not found.",
                "message": "The specified model does not exist."
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Unexpected error in prediction for model {pk}: {str(e)}")
            return Response({
                "error": "An unexpected error occurred.",
                "message": "Please try again later."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def put(self, request, pk):
        try:
            trained_model = self.get_object(pk)
            
            if hasattr(trained_model, 'user_id') and trained_model.user_id != request.user.id:
                return Response({
                    "error": "Permission denied.",
                    "message": "You can only modify your own models."
                }, status=status.HTTP_403_FORBIDDEN)
            
            data = {
                'is_public': not trained_model.is_public
            }
            
            serializer = TrainedModelSerializer(trained_model, data=data, partial=True)
            if serializer.is_valid():
                serializer.save()
                
                if trained_model.model_file:
                    try:
                        model_file_path = trained_model.model_file.path
                        
                        if not os.path.exists(model_file_path):
                            logger.warning(f"Model file not found: {model_file_path}")
                        else:
                            model = joblib.load(model_file_path)
                            model_coefficients = model.coef_.tolist() if hasattr(model, "coef_") else None
                            model_intercept = model.intercept_.tolist() if hasattr(model, "intercept_") else None
                            
                    except (FileNotFoundError, EOFError, ValueError) as e:
                        logger.error(f"Error loading model file for model {pk}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Unexpected error loading model file for model {pk}: {str(e)}")
                    
                return Response({
                    "message": f"Model visibility updated to {'public' if data['is_public'] else 'private'}.",
                    "model": {
                        "id": str(trained_model.id),
                        "user_id": str(trained_model.user_id),
                        "name": trained_model.model_name,
                        "type": trained_model.model_type,
                        "polynomial_degree": trained_model.polynomial_degree,
                        "target_column": trained_model.target_column,
                        "features": trained_model.features,
                        "is_public": trained_model.is_public,
                        "likes": trained_model.likes,
                        "created_at": trained_model.created_at,
                        "model_file": trained_model.model_file.url if trained_model.model_file else None,
                    },
                    "metrics": ModelStatsSerializer(trained_model.stats).data if hasattr(trained_model, "stats") else {},
                    "graphs": ModelGraphSerializer(trained_model.graphs.all(), many=True).data,
                    "coefficients": model_coefficients,
                    "intercept": model_intercept,
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    "error": "Validation failed.",
                    "message": "Invalid data provided.",
                    "details": serializer.errors
                }, status=status.HTTP_400_BAD_REQUEST)
                
        except Http404:
            return Response({
                "error": "Model not found.",
                "message": "The specified model does not exist."
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error updating model {pk}: {str(e)}")
            return Response({
                "error": "An error occurred while updating the model.",
                "message": "Please try again later."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ModelUpdateView(APIView):
    
    permission_classes = [IsAuthenticated]
    
    def get_object(self, pk):
        try:
            return TrainedModel.objects.get(pk=pk)
        except TrainedModel.DoesNotExist:
            raise Http404("Model not found.")
        except ValidationError:
            raise Http404("Invalid model ID format.")
    
    def put(self, request, pk):
        try:
            trained_model = self.get_object(pk)
            
            state = request.data.get('state')
            if not state:
                return Response({
                    "error": "Missing required field.",
                    "message": "'state' field is required."
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if state not in ['like', 'dislike']:
                return Response({
                    "error": "Invalid state value.",
                    "message": "'state' must be either 'like' or 'dislike'."
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if state == 'like':
                trained_model.likes += 1
            elif state == 'dislike':
                if trained_model.likes > 0:
                    trained_model.likes -= 1
                else:
                    return Response({
                        "error": "Invalid operation.",
                        "message": "Cannot dislike a model with zero likes."
                    }, status=status.HTTP_400_BAD_REQUEST)
            
            trained_model.save()
            serializer = TrainedModelSerializer(trained_model)
            
            return Response({
                "message": f"Model {state}d successfully.",
                "data": serializer.data,
                "state": state
            }, status=status.HTTP_200_OK)
            
        except Http404:
            return Response({
                "error": "Model not found.",
                "message": "The specified model does not exist."
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            logger.error(f"Error updating likes for model {pk}: {str(e)}")
            return Response({
                "error": "An error occurred while updating the model.",
                "message": "Please try again later."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
  
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def getUserLikedModels(request):
    try:
        if not request.body:
            return Response({
                "error": "Empty request body.",
                "message": "Request body with 'models' field is required."
            }, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError as e:
            return Response({
                "error": "Invalid JSON format.",
                "message": "Please provide valid JSON data.",
                "details": str(e)
            }, status=status.HTTP_400_BAD_REQUEST)

        model_ids = data.get('models')
        
        if len(model_ids) == 0:
            return Response({
                "message": "No models requested.",
                "data": []
            }, status=status.HTTP_200_OK)

        try:
            liked_models = TrainedModel.objects.filter(id__in=model_ids)
            serializer = TrainedModelSerializer(liked_models, many=True)
            
            return Response({
                "message": "Liked models retrieved successfully.",
                "data": serializer.data,
                "total_found": len(serializer.data),
                "total_requested": len(model_ids)
            }, status=status.HTTP_200_OK)
            
        except ValidationError as e:
            return Response({
                "error": "Invalid model ID format.",
                "message": "One or more model IDs have invalid format.",
                "details": str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        logger.error(f"Unexpected error in getUserLikedModels: {str(e)}")
        return Response({
            "error": "An unexpected error occurred.",
            "message": "Please try again later."
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
def download_model_report(request, model_id):
    model = get_object_or_404(TrainedModel, id=model_id)
    
    generator = ModelReportGenerator(model)
    pdf_content = generator.generate_report()
    
    response = HttpResponse(pdf_content, content_type='application/pdf')
    filename = f"{model.model_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    return response

logger = logging.getLogger(__name__)

class TrainModelView(APIView):
    parser_classes = [MultiPartParser]
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user = request.user

        try:
            curr_user = User.objects.filter(id=user.id).first()
            if not curr_user:
                return Response({"message": "User not found."}, status=status.HTTP_404_NOT_FOUND)

            if not curr_user.premium_user and curr_user.limit <= 0:
                return Response({
                    "message": "You have reached your limit. Please upgrade to premium.",
                    "success": False
                }, status=status.HTTP_403_FORBIDDEN)

            model_name = request.data.get("model_name")
            target_col = request.data.get("target_col")
            endpoint = request.data.get("endpoint")
            file = request.FILES.get("csv_file")

            if not all([model_name, target_col, endpoint, file]):
                return Response({
                    "message": "Missing required fields: model_name, target_col, endpoint, or file.",
                    "success": False
                }, status=status.HTTP_400_BAD_REQUEST)

            files = {
                "csv_file": (file.name, file.read(), file.content_type)
            }
            data = {
                "model_name": model_name,
                "target_col": target_col
            }

            token = request.META.get("HTTP_AUTHORIZATION")
            if not token:
                return Response({"message": "Authorization token missing."}, status=status.HTTP_401_UNAUTHORIZED)

            token = token.replace("Bearer ", "")

            response = requests.post(
                f"http://localhost:8000/api/v1/{endpoint}/",
                data=data,
                files=files,
                headers={"Authorization": f"Bearer {token}"}
            )

            if response.status_code == 200:
                if not curr_user.premium_user:
                    curr_user.limit -= 1
                    curr_user.save()
                return Response({
                    "message": "Model training completed.",
                    "success": True,
                    "data": response.json()
                }, status=status.HTTP_200_OK)
            else:
                return Response({
                    "message": "Model training failed at ML server.",
                    "success": False,
                    "error": response.json()
                }, status=response.status_code)

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during model training: {str(e)}")
            return Response({
                "message": "Error communicating with ML server.",
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.exception("Internal server error")
            return Response({
                "message": "Internal server error.",
                "success": False,
                "error": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)