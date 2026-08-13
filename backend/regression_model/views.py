import logging
from django.core.files import File
from django.core.files.images import ImageFile
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import joblib
import os
from tempfile import NamedTemporaryFile

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

class LinearRegressionView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            data = request.data
            model_name = data.get('model_name', 'Linear Regression')
            target_col = data.get('target_col')

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

            # Validate target column is numeric for regression
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
                model = LinearRegression()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
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
                joblib.dump(model, temp_file.name)
                with open(temp_file.name, 'rb') as f:
                    django_file = File(f)
                    ml_model = TrainedModel.objects.create(
                        model_type=TrainedModel.ModelType.LINEAR_REGRESSION,
                        model_name=model_name,
                        target_column=target_col,
                        features=",".join(X.columns),
                        user_id=request.user.id
                    )
                    ml_model.model_file.save(f"{ml_model.id}_model.pkl", django_file)
                    ml_model.csv_file.save(f"{ml_model.id}_data.csv", csv_file)
                    ml_model.save()
                # os.remove(temp_file.name)
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
                    model.coef_.tolist() if hasattr(model, "coef_") else None
                ),
                "intercept": (
                    model.intercept_.tolist() if hasattr(model, "intercept_") else None
                ),
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Unexpected error in LinearRegressionView.post", exc_info=True)
            return Response({
                "error": "Unexpected server error occurred.",
                "code": "UNEXPECTED_ERROR",
                "details": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PolynomialRegressionView(APIView):
    permission_classes = [IsAuthenticated]

    def polynomial_degree_trainer(self, degree, x_train, y_train, x_test, y_test):
        try:
            poly = PolynomialFeatures(degree=degree)
            model = LinearRegression()
            pipeline = make_pipeline(poly, model)
            pipeline.fit(x_train, y_train)
            y_pred = pipeline.predict(x_test)
            metrics = calculate_regression_metrics(y_test, y_pred)
            return pipeline, metrics
        except Exception as e:
            logger.warning(f"Error training polynomial degree {degree}: {str(e)}")
            return None, None

    def post(self, request):
        try:
            data = request.data
            model_name = data.get('model_name', 'Polynomial Regression')
            target_col = data.get('target_col')

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

            # Validate target column is numeric for regression
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

            best_r2 = float('-inf')
            best_degree = 0
            best_metrics = {}
            best_pipeline = None

            try:
                for degree in range(1, 10):
                    pipeline, metrics = self.polynomial_degree_trainer(degree, x_train, y_train, x_test, y_test)
                    if pipeline is not None and metrics is not None:
                        if metrics['r2_score'] > best_r2:
                            best_r2 = metrics['r2_score']
                            best_degree = degree
                            best_metrics = metrics
                            best_pipeline = pipeline

                if best_pipeline is None:
                    return Response({
                        "error": "Failed to train any polynomial model.",
                        "code": "POLYNOMIAL_TRAINING_FAILED"
                    }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            except Exception as e:
                logger.error(f"Error during polynomial degree selection: {str(e)}")
                return Response({
                    "error": f"Failed during model training: {str(e)}",
                    "code": "MODEL_TRAINING_ERROR"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            try:
                temp_file = NamedTemporaryFile(delete=False, suffix=".pkl")
                joblib.dump(best_pipeline, temp_file.name)
                with open(temp_file.name, 'rb') as f:
                    django_file = File(f)
                    ml_model = TrainedModel.objects.create(
                        model_type=TrainedModel.ModelType.POLYNOMIAL_REGRESSION,
                        model_name=model_name,
                        polynomial_degree=best_degree,
                        target_column=target_col,
                        features=",".join(X.columns),
                        user_id=request.user.id
                    )
                    ml_model.model_file.save(f"{ml_model.id}_model.pkl", django_file)
                    ml_model.csv_file.save(f"{ml_model.id}_data.csv", csv_file)
                    ml_model.save()
                # os.remove(temp_file.name)
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
                return Response({
                    "error": f"Failed to save model: {str(e)}",
                    "code": "MODEL_SAVE_ERROR"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            try:
                ModelStats.objects.create(
                    trained_model=ml_model,
                    r2_score=best_metrics["r2_score"],
                    mse=best_metrics["mse"],
                    mae=best_metrics["mae"]
                )
            except Exception as e:
                logger.warning(f"Error saving stats: {str(e)}")

            try:
                y_pred = best_pipeline.predict(x_test)

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

            try:
                linear_model = best_pipeline.named_steps['linearregression']
                coefficients = linear_model.coef_.tolist() if hasattr(linear_model, "coef_") else None
                intercept = linear_model.intercept_.tolist() if hasattr(linear_model, "intercept_") else None
            except Exception as e:
                logger.warning(f"Error extracting model coefficients: {str(e)}")
                coefficients = None
                intercept = None

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
                "coefficients": coefficients,
                "intercept": intercept,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Unexpected error in PolynomialRegressionView.post", exc_info=True)
            return Response({
                "error": "Unexpected server error occurred.",
                "code": "UNEXPECTED_ERROR",
                "details": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)