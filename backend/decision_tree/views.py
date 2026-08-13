from django.core.files import File
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
import pandas as pd
import numpy as np

import joblib
from tempfile import NamedTemporaryFile
import os
import logging

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, f1_score, roc_curve, auc, precision_recall_curve
)
from django.core.files.images import ImageFile
from trained_model.models import TrainedModel, ModelStats, ModelGraph
from ml_utils.graph_utils import *
from ml_utils.stats_utils import *

from trained_model.serializer import ModelStatsSerializer, ModelGraphSerializer

# Set up logging
logger = logging.getLogger(__name__)

class DecisionTreeView(APIView):
    
    permission_classes = [IsAuthenticated]
    
    def hyperparameter_tuning(self, x_train, y_train, x_test, y_test):
        """
        Perform hyperparameter tuning for Decision Tree
        """
        try:
            best_score = 0
            best_params = {}
            
            for max_depth in range(1, 11):
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                model.fit(x_train, y_train)
                
                y_pred = model.predict(x_test)
                score = accuracy_score(y_test, y_pred)
                if score > best_score:
                    best_score = score
                    best_params = {'max_depth': max_depth, 'model': model}
            
            if not best_params:
                raise ValueError("No valid hyperparameters found during tuning")
                
            logger.info(f"Best params: {best_params}, Best score: {best_score}")
            return best_params, best_score
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            raise
    
    def post(self, request):
        try:
            # Validate request data
            if not request.data:
                return Response({
                    "error": "No data provided in request",
                    "code": "MISSING_DATA"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            data = request.data
            model_name = data.get('model_name', 'Decision Tree')
            target_col = data.get('target_col')
            
            # Validate required fields
            if not target_col:
                return Response({
                    "error": "Target column is required",
                    "code": "MISSING_TARGET_COLUMN"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate CSV file
            if 'csv_file' not in request.FILES:
                return Response({
                    "error": "CSV file is required",
                    "code": "MISSING_CSV_FILE"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            csv_file = request.FILES['csv_file']
            
            # Validate file type
            if not csv_file.name.lower().endswith('.csv'):
                return Response({
                    "error": "File must be a CSV file",
                    "code": "INVALID_FILE_TYPE"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Read CSV file
            try:
                df = pd.read_csv(csv_file)
            except pd.errors.EmptyDataError:
                return Response({
                    "error": "CSV file is empty",
                    "code": "EMPTY_CSV_FILE"
                }, status=status.HTTP_400_BAD_REQUEST)
            except pd.errors.ParserError as e:
                return Response({
                    "error": f"Error parsing CSV file: {str(e)}",
                    "code": "CSV_PARSE_ERROR"
                }, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({
                    "error": f"Error reading CSV file: {str(e)}",
                    "code": "CSV_READ_ERROR"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate DataFrame
            if df.empty:
                return Response({
                    "error": "CSV file contains no data",
                    "code": "EMPTY_DATASET"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if len(df) < 10:
                return Response({
                    "error": "Dataset must contain at least 10 rows for meaningful analysis",
                    "code": "INSUFFICIENT_DATA"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate target column
            if target_col not in df.columns:
                return Response({
                    "error": f"Target column '{target_col}' not found in CSV file",
                    "code": "TARGET_COLUMN_NOT_FOUND",
                    "available_columns": list(df.columns)
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Check for null values
            if df.isnull().values.any() and df.isnull().sum().sum() > 50:
                return Response({
                    "error": "Dataset contains too many null values (>50)",
                    "code": "TOO_MANY_NULLS",
                    "null_count": int(df.isnull().sum().sum())
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Drop null values
            original_length = len(df)
            df = df.dropna()
            
            if df.empty:
                return Response({
                    "error": "No data remaining after removing null values",
                    "code": "NO_DATA_AFTER_CLEANING"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if len(df) < 5:
                return Response({
                    "error": "Insufficient data after cleaning (less than 5 rows)",
                    "code": "INSUFFICIENT_DATA_AFTER_CLEANING"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Validate target column values
            y_raw = df[target_col]
            unique_targets = y_raw.nunique()
            
            if unique_targets < 2:
                return Response({
                    "error": "Target column must have at least 2 unique values for classification",
                    "code": "INSUFFICIENT_TARGET_CLASSES",
                    "unique_values": int(unique_targets)
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if unique_targets > 50:
                return Response({
                    "error": "Target column has too many unique values (>50). Consider using regression instead.",
                    "code": "TOO_MANY_TARGET_CLASSES",
                    "unique_values": int(unique_targets)
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Encode target variable
            try:
                y = y_raw.astype(int)
            except (ValueError, TypeError):
                try:
                    le = LabelEncoder()
                    y = le.fit_transform(y_raw)
                except Exception as e:
                    return Response({
                        "error": f"Error encoding target variable: {str(e)}",
                        "code": "TARGET_ENCODING_ERROR"
                    }, status=status.HTTP_400_BAD_REQUEST)
            
            # Prepare features
            try:
                X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
            except Exception as e:
                return Response({
                    "error": f"Error preparing feature matrix: {str(e)}",
                    "code": "FEATURE_PREPARATION_ERROR"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            if X.empty or X.shape[1] == 0:
                return Response({
                    "error": "No features available after preprocessing",
                    "code": "NO_FEATURES_AVAILABLE"
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # Split data
            try:
                x_train, x_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError as e:
                # Try without stratification if it fails
                try:
                    x_train, x_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                except Exception as e:
                    return Response({
                        "error": f"Error splitting data: {str(e)}",
                        "code": "DATA_SPLIT_ERROR"
                    }, status=status.HTTP_400_BAD_REQUEST)
            
            # Hyperparameter tuning
            try:
                best_params, best_score = self.hyperparameter_tuning(x_train, y_train, x_test, y_test)
            except Exception as e:
                return Response({
                    "error": f"Error during hyperparameter tuning: {str(e)}",
                    "code": "HYPERPARAMETER_TUNING_ERROR"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Save model
            try:
                temp_file = NamedTemporaryFile(delete=False, suffix=".pkl")
                joblib.dump(best_params['model'], temp_file.name)
                
                with open(temp_file.name, 'rb') as f:
                    django_file = File(f)
                    ml_model = TrainedModel.objects.create(
                        model_type=TrainedModel.ModelType.DECISION_TREE,
                        model_name=model_name,
                        target_column=target_col,
                        features=",".join(X.columns),
                        user_id=request.user.id
                    )
                    ml_model.model_file.save(f"{ml_model.id}_model.pkl", django_file)
                    ml_model.csv_file.save(f"{ml_model.id}_data.csv", csv_file)
                    ml_model.save()
                
            except Exception as e:
                try:
                    if 'temp_file' in locals():
                        os.unlink(temp_file.name)
                except:
                    pass
                
                return Response({
                    "error": f"Error saving model: {str(e)}",
                    "code": "MODEL_SAVE_ERROR"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # Calculate model statistics
            try:
                y_pred = best_params['model'].predict(x_test)
                
                # Check if it's binary or multiclass
                n_classes = len(np.unique(y))
                is_binary = n_classes == 2
                
                # Get probabilities for ROC/PR curves
                y_proba = None
                if hasattr(best_params['model'], 'predict_proba'):
                    y_proba_full = best_params['model'].predict_proba(x_test)
                    if is_binary:
                        y_proba = y_proba_full[:, 1]
                    else:
                        y_proba = y_proba_full
                
                avg_method = 'binary' if is_binary else 'macro'
                
                ModelStats.objects.create(
                    trained_model=ml_model,
                    accuracy=accuracy_score(y_test, y_pred),
                    precision=precision_score(y_test, y_pred, average=avg_method, zero_division=0),
                    recall=recall_score(y_test, y_pred, average=avg_method, zero_division=0),
                    f1_score=f1_score(y_test, y_pred, average=avg_method, zero_division=0),
                )
                
            except Exception as e:
                logger.error(f"Error calculating model statistics: {str(e)}")
            
            # Generate and save graphs
            try:
                graph_dir = 'media/graphs'
                os.makedirs(graph_dir, exist_ok=True)
                
                # 1. Confusion Matrix
                try:
                    cm_path = os.path.join(graph_dir, f'cm_{ml_model.id}.png')
                    save_confusion_matrix_graph(y_test, y_pred, cm_path)
                    with open(cm_path, 'rb') as img_file:
                        ModelGraph.objects.create(
                            trained_model=ml_model,
                            title="Confusion Matrix",
                            description="Shows TP, FP, FN, TN",
                            graph_image=ImageFile(img_file, name=os.path.basename(cm_path))
                        )
                except Exception as e:
                    logger.error(f"Error creating confusion matrix: {str(e)}")
                
                # 2. ROC and PR Curves
                if y_proba is not None:
                    try:
                        if is_binary:
                            # ROC Curve
                            roc_path = os.path.join(graph_dir, f'roc_{ml_model.id}.png')
                            save_roc_curve_graph(y_test, y_proba, roc_path)
                            with open(roc_path, 'rb') as img_file:
                                ModelGraph.objects.create(
                                    trained_model=ml_model,
                                    title="ROC Curve",
                                    description="Shows model's ability to distinguish classes",
                                    graph_image=ImageFile(img_file, name=os.path.basename(roc_path))
                                )
                            
                            # Precision-Recall Curve
                            pr_path = os.path.join(graph_dir, f'pr_{ml_model.id}.png')
                            save_precision_recall_graph(y_test, y_proba, pr_path)
                            with open(pr_path, 'rb') as img_file:
                                ModelGraph.objects.create(
                                    trained_model=ml_model,
                                    title="Precision-Recall Curve",
                                    description="Shows trade-off between precision and recall",
                                    graph_image=ImageFile(img_file, name=os.path.basename(pr_path))
                                )
                        else:
                            # Multiclass ROC
                            roc_path = os.path.join(graph_dir, f'roc_{ml_model.id}.png')
                            save_multiclass_roc_curve_graph(y_test, y_proba, roc_path)
                            with open(roc_path, 'rb') as img_file:
                                ModelGraph.objects.create(
                                    trained_model=ml_model,
                                    title="ROC Curve (Multiclass)",
                                    description="Shows model's ability to distinguish between multiple classes",
                                    graph_image=ImageFile(img_file, name=os.path.basename(roc_path))
                                )
                            
                            # Multiclass Precision-Recall
                            pr_path = os.path.join(graph_dir, f'pr_{ml_model.id}.png')
                            save_multiclass_precision_recall_graph(y_test, y_proba, pr_path)
                            with open(pr_path, 'rb') as img_file:
                                ModelGraph.objects.create(
                                    trained_model=ml_model,
                                    title="Precision-Recall Curve (Multiclass)",
                                    description="Shows precision-recall trade-off for multiple classes",
                                    graph_image=ImageFile(img_file, name=os.path.basename(pr_path))
                                )
                    except Exception as e:
                        logger.error(f"Error creating ROC/PR curves: {str(e)}")
                        
            except Exception as e:
                logger.error(f"Error creating graphs: {str(e)}")
            
            # Return successful response
            try:
                return Response({
                    "message": "Model, statistics, and graphs saved successfully.",
                    "model": {
                        "id": str(ml_model.id),
                        "name": ml_model.model_name,
                        "type": ml_model.model_type,
                        "polynomial_degree": ml_model.polynomial_degree,
                        "target_column": ml_model.target_column,
                        "features": ml_model.features,
                        "is_public": ml_model.is_public,
                        "likes": ml_model.likes,
                        "created_at": ml_model.created_at,
                    },
                    "metrics": ModelStatsSerializer(ml_model.stats).data if hasattr(ml_model, "stats") else {},
                    "graphs": ModelGraphSerializer(ml_model.graphs.all(), many=True).data,
                    "coefficients": (
                        ml_model.coef_.tolist() if hasattr(ml_model, "coef_") else None
                    ),
                    "intercept": (
                        ml_model.intercept_.tolist() if hasattr(ml_model, "intercept_") else None
                    ),
                }, status=status.HTTP_200_OK)
                
            except Exception as e:
                return Response({
                    "error": f"Error serializing response: {str(e)}",
                    "code": "RESPONSE_SERIALIZATION_ERROR"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        except Exception as e:
            logger.error(f"Unexpected error in DecisionTreeView.post: {str(e)}", exc_info=True)
            return Response({
                "error": "An unexpected error occurred while processing your request",
                "code": "UNEXPECTED_ERROR",
                "details": str(e) if hasattr(e, '__str__') else "Unknown error"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)