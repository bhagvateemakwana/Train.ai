import logging
from django.core.files import File
from django.core.files.images import ImageFile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import joblib
import os
from tempfile import NamedTemporaryFile
import numpy as np

from trained_model.models import TrainedModel, ModelStats, ModelGraph
from ml_utils.graph_utils import (
    save_residual_plot,
    save_actual_vs_predicted_plot,
    save_error_distribution_plot,
    save_qq_plot
)
from ml_utils.stats_utils import calculate_regression_metrics
from trained_model.serializer import ModelStatsSerializer, ModelGraphSerializer

logger = logging.getLogger(__name__)

class RidgeRegressionView(APIView):
    permission_classes = [IsAuthenticated]

    def find_best_alpha(self, x_train, y_train, cv_folds=5):
        try:
            alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
            
            pipeline = make_pipeline(StandardScaler(), Ridge())
            
            param_grid = {'ridge__alpha': alphas}
            grid_search = GridSearchCV(
                pipeline, 
                param_grid, 
                cv=cv_folds, 
                scoring='r2',
                n_jobs=-1
            )
            
            grid_search.fit(x_train, y_train)
            
            return grid_search.best_params_['ridge__alpha'], grid_search.best_estimator_
        except Exception as e:
            logger.warning(f"Error in grid search, using default alpha: {str(e)}")
            # Fallback to default pipeline
            pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            pipeline.fit(x_train, y_train)
            return 1.0, pipeline

    def post(self, request):
        try:
            data = request.data
            model_name = data.get('model_name', 'Ridge Regression')
            target_col = data.get('target_col')
            custom_alpha = data.get('alpha') 
            
            if not target_col:
                return Response({
                    "error": "Target column is required.",
                    "code": "MISSING_TARGET_COLUMN"
                }, status=status.HTTP_400_BAD_REQUEST)

            if 'csv_file' not in request.FILES:
                return Response({
                    "error": "CSV file is required.",
                    "code": "MISSING_CSV_FILE"
                }, status=status.HTTP_400_BAD_REQUEST)

            csv_file = request.FILES['csv_file']

            if not csv_file.name.lower().endswith('.csv'):
                return Response({
                    "error": "File must be a CSV.",
                    "code": "INVALID_FILE_TYPE"
                }, status=status.HTTP_400_BAD_REQUEST)

            try:
                df = pd.read_csv(csv_file)
            except pd.errors.EmptyDataError:
                return Response({
                    "error": "CSV file is empty.",
                    "code": "EMPTY_CSV_FILE"
                }, status=status.HTTP_400_BAD_REQUEST)
            except pd.errors.ParserError as e:
                return Response({
                    "error": f"CSV parse error: {str(e)}",
                    "code": "CSV_PARSE_ERROR"
                }, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({
                    "error": f"Error reading CSV: {str(e)}",
                    "code": "CSV_READ_ERROR"
                }, status=status.HTTP_400_BAD_REQUEST)

            if df.empty:
                return Response({
                    "error": "CSV contains no data.",
                    "code": "EMPTY_DATASET"
                }, status=status.HTTP_400_BAD_REQUEST)

            if target_col not in df.columns:
                return Response({
                    "error": f"Target column '{target_col}' not found.",
                    "code": "TARGET_COLUMN_NOT_FOUND",
                    "available_columns": list(df.columns)
                }, status=status.HTTP_400_BAD_REQUEST)

            if df.isnull().sum().sum() > 50:
                return Response({
                    "error": "Dataset has too many null values.",
                    "code": "TOO_MANY_NULLS"
                }, status=status.HTTP_400_BAD_REQUEST)

            df = df.dropna()
            if df.empty or len(df) < 5:
                return Response({
                    "error": "Insufficient data after cleaning.",
                    "code": "INSUFFICIENT_DATA"
                }, status=status.HTTP_400_BAD_REQUEST)

            try:
                y = pd.to_numeric(df[target_col], errors='coerce')
                if y.isnull().any():
                    return Response({
                        "error": "Target column must contain numeric values for regression.",
                        "code": "NON_NUMERIC_TARGET"
                    }, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                return Response({
                    "error": f"Error processing target column: {str(e)}",
                    "code": "TARGET_PROCESSING_ERROR"
                }, status=status.HTTP_400_BAD_REQUEST)

            try:
                X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
            except Exception as e:
                return Response({
                    "error": f"Error creating features: {str(e)}",
                    "code": "FEATURE_PREPARATION_ERROR"
                }, status=status.HTTP_400_BAD_REQUEST)

            if X.empty or X.shape[1] == 0:
                return Response({
                    "error": "No features after preprocessing.",
                    "code": "NO_FEATURES"
                }, status=status.HTTP_400_BAD_REQUEST)

            try:
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            except Exception as e:
                return Response({
                    "error": f"Error splitting data: {str(e)}",
                    "code": "DATA_SPLIT_ERROR"
                }, status=status.HTTP_400_BAD_REQUEST)

            try:
                if custom_alpha is not None:
                    try:
                        alpha_value = float(custom_alpha)
                        if alpha_value <= 0:
                            return Response({
                                "error": "Alpha must be a positive number.",
                                "code": "INVALID_ALPHA_VALUE"
                            }, status=status.HTTP_400_BAD_REQUEST)
                        
                        model_pipeline = make_pipeline(StandardScaler(), Ridge(alpha=alpha_value))
                        model_pipeline.fit(x_train, y_train)
                        best_alpha = alpha_value
                    except ValueError:
                        return Response({
                            "error": "Alpha must be a valid number.",
                            "code": "INVALID_ALPHA_FORMAT"
                        }, status=status.HTTP_400_BAD_REQUEST)
                else:
                    best_alpha, model_pipeline = self.find_best_alpha(x_train, y_train)
                
                y_pred = model_pipeline.predict(x_test)
                
            except Exception as e:
                logger.error(f"Model training error: {str(e)}")
                return Response({
                    "error": f"Failed to train model: {str(e)}",
                    "code": "MODEL_TRAINING_ERROR"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            try:
                metrics = calculate_regression_metrics(y_test, y_pred)
            except Exception as e:
                logger.warning(f"Error calculating metrics: {str(e)}")
                try:
                    metrics = {
                        "r2_score": r2_score(y_test, y_pred),
                        "mse": mean_squared_error(y_test, y_pred),
                        "mae": mean_absolute_error(y_test, y_pred)
                    }
                except Exception:
                    metrics = {"r2_score": 0, "mse": 0, "mae": 0}

            try:
                temp_file = NamedTemporaryFile(delete=False, suffix=".pkl")
                joblib.dump(model_pipeline, temp_file.name)
                with open(temp_file.name, 'rb') as f:
                    django_file = File(f)
                    ml_model = TrainedModel.objects.create(
                        model_type=TrainedModel.ModelType.RIDGE_REGRESSION,
                        model_name=model_name,
                        target_column=target_col,
                        features=",".join(X.columns),
                        user_id=request.user.id,
                        # alpha_value=best_alpha if hasattr(TrainedModel, 'alpha_value') else None
                    )
                    ml_model.model_file.save(f"{ml_model.id}_model.pkl", django_file)
                    ml_model.csv_file.save(f"{ml_model.id}_data.csv", csv_file)
                    ml_model.save()
                
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                return Response({
                    "error": f"Failed to save model: {str(e)}",
                    "code": "MODEL_SAVE_ERROR"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            try:
                ModelStats.objects.create(
                    trained_model=ml_model,
                    r2_score=metrics["r2_score"],
                    mse=metrics["mse"],
                    mae=metrics["mae"]
                )
            except Exception as e:
                logger.warning(f"Error saving stats: {str(e)}")

            try:
                graph_dir = 'media/graphs'
                os.makedirs(graph_dir, exist_ok=True)

                residual_path = os.path.join(graph_dir, f"residual_{ml_model.id}.png")
                pred_path = os.path.join(graph_dir, f"pred_{ml_model.id}.png")
                err_dist_path = os.path.join(graph_dir, f"err_dist_{ml_model.id}.png")
                qq_path = os.path.join(graph_dir, f"qq_{ml_model.id}.png")

                save_residual_plot(y_test, y_pred, residual_path)
                save_actual_vs_predicted_plot(y_test, y_pred, pred_path)
                save_error_distribution_plot(y_test, y_pred, err_dist_path)
                save_qq_plot(y_test, y_pred, qq_path)

                for path, title, desc in [
                    (residual_path, "Residual Plot", "Shows residuals vs predictions"),
                    (pred_path, "Actual vs Predicted", "Compares predicted vs actual values"),
                    (err_dist_path, "Error Distribution", "Distribution of prediction errors"),
                    (qq_path, "Q-Q Plot", "Check if residuals are normally distributed")
                ]:
                    with open(path, 'rb') as img_file:
                        ModelGraph.objects.create(
                            trained_model=ml_model,
                            title=title,
                            description=desc,
                            graph_image=ImageFile(img_file, name=os.path.basename(path))
                        )
            except Exception as e:
                logger.warning(f"Error generating graphs: {str(e)}")

            # Extract coefficients from the Ridge model in the pipeline
            try:
                ridge_model = model_pipeline.named_steps['ridge']
                coefficients = ridge_model.coef_.tolist() if hasattr(ridge_model, "coef_") else None
                intercept = ridge_model.intercept_.tolist() if hasattr(ridge_model, "intercept_") else None
            except Exception as e:
                logger.warning(f"Error extracting model coefficients: {str(e)}")
                coefficients = None
                intercept = None

            return Response({
                "message": "Ridge regression model, statistics, and graphs saved successfully.",
                "model": {
                    "id": str(ml_model.id),
                    "name": ml_model.model_name,
                    "type": ml_model.model_type,
                    "alpha": best_alpha,
                    "target_column": ml_model.target_column,
                    "features": ml_model.features,
                    "is_public": ml_model.is_public,
                    "likes": ml_model.likes,
                    "created_at": ml_model.created_at,
                },
                "metrics": ModelStatsSerializer(ml_model.stats).data if hasattr(ml_model, "stats") else {},
                "graphs": ModelGraphSerializer(ml_model.graphs.all(), many=True).data,
                "coefficients": coefficients,
                "intercept": intercept,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Unexpected error in RidgeRegressionView.post", exc_info=True)
            return Response({
                "error": "Unexpected server error occurred.",
                "code": "UNEXPECTED_ERROR",
                "details": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)