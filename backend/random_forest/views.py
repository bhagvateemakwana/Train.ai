import logging
from django.core.files import File
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import joblib
from tempfile import NamedTemporaryFile
import os
from django.core.files.images import ImageFile
from sklearn.preprocessing import LabelEncoder
import numpy as np

from trained_model.models import TrainedModel, ModelStats, ModelGraph
from ml_utils.graph_utils import *
from ml_utils.stats_utils import *
from trained_model.serializer import ModelStatsSerializer, ModelGraphSerializer

logger = logging.getLogger(__name__)

class RandomForestView(APIView):
    permission_classes = [IsAuthenticated]

    def hyperparameter_tuning(self, x_train, y_train, x_test, y_test):
        best_score = 0
        best_params = {}
        
        n_estimators_options = [50, 100, 200]
        max_depth_options = [None, 5, 10, 15]
        min_samples_split_options = [2, 5, 10]
        min_samples_leaf_options = [1, 2, 4]
        
        for n_estimators in n_estimators_options:
            for max_depth in max_depth_options:
                for min_samples_split in min_samples_split_options:
                    for min_samples_leaf in min_samples_leaf_options:
                        try:
                            model = RandomForestClassifier(
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42,
                                n_jobs=-1
                            )
                            model.fit(x_train, y_train)
                            y_pred = model.predict(x_test)
                            score = accuracy_score(y_test, y_pred)
                            
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'model': model
                                }
                            logger.info(f"Best Params After n_estimators:{n_estimators}, max_depth:{max_depth}, min_samples_split:{min_samples_split}, min_samples_leaf:{min_samples_leaf}")
                        except Exception as e:
                            logger.warning(f"Error in hyperparameter tuning: {str(e)}")
                            continue
        
        if not best_params:
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            best_score = accuracy_score(y_test, y_pred)
            best_params = {'model': model}
            
        return best_params, best_score

    def post(self, request):
        try:
            data = request.data
            model_name = data.get('model_name', 'Random Forest')
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

            y_raw = df[target_col]
            try:
                y = y_raw.astype(int)
            except (ValueError, TypeError):
                try:
                    le = LabelEncoder()
                    y = le.fit_transform(y_raw)
                except Exception as e:
                    return Response({
                        "error": f"Error encoding target: {str(e)}",
                        "code": "TARGET_ENCODING_ERROR"
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
                x_train, x_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except Exception:
                x_train, x_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            best_params, _ = self.hyperparameter_tuning(x_train, y_train, x_test, y_test)
            model = best_params['model']

            try:
                temp_file = NamedTemporaryFile(delete=False, suffix=".pkl")
                joblib.dump(model, temp_file.name)
                with open(temp_file.name, 'rb') as f:
                    django_file = File(f)
                    ml_model = TrainedModel.objects.create(
                        model_type=TrainedModel.ModelType.RANDOM_FOREST,
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
                y_pred = model.predict(x_test)
                y_proba = model.predict_proba(x_test)[:, 1] if model.predict_proba(x_test).shape[1] == 2 else None

                ModelStats.objects.create(
                    trained_model=ml_model,
                    accuracy=accuracy_score(y_test, y_pred),
                    precision=precision_score(y_test, y_pred, average='macro', zero_division=0),
                    recall=recall_score(y_test, y_pred, average='macro', zero_division=0),
                    f1_score=f1_score(y_test, y_pred, average='macro', zero_division=0),
                )
            except Exception as e:
                logger.warning(f"Error calculating stats: {str(e)}")

            try:
                graph_dir = 'media/graphs'
                os.makedirs(graph_dir, exist_ok=True)

                cm_path = os.path.join(graph_dir, f'cm_{ml_model.id}.png')
                save_confusion_matrix_graph(y_test, y_pred, cm_path)
                with open(cm_path, 'rb') as img_file:
                    ModelGraph.objects.create(
                        trained_model=ml_model,
                        title="Confusion Matrix",
                        description="Shows TP, FP, FN, TN",
                        graph_image=ImageFile(img_file, name=os.path.basename(cm_path))
                    )

                if y_proba is not None:
                    roc_path = os.path.join(graph_dir, f'roc_{ml_model.id}.png')
                    save_roc_curve_graph(y_test, y_proba, roc_path)
                    with open(roc_path, 'rb') as img_file:
                        ModelGraph.objects.create(
                            trained_model=ml_model,
                            title="ROC Curve",
                            description="Shows ability to distinguish classes",
                            graph_image=ImageFile(img_file, name=os.path.basename(roc_path))
                        )

                    pr_path = os.path.join(graph_dir, f'pr_{ml_model.id}.png')
                    save_precision_recall_graph(y_test, y_proba, pr_path)
                    with open(pr_path, 'rb') as img_file:
                        ModelGraph.objects.create(
                            trained_model=ml_model,
                            title="Precision-Recall Curve",
                            description="Shows trade-off between precision and recall",
                            graph_image=ImageFile(img_file, name=os.path.basename(pr_path))
                        )

                try:
                    feature_importance_path = os.path.join(graph_dir, f'feature_importance_{ml_model.id}.png')
                    save_feature_importance_graph(model.feature_importances_, X.columns, feature_importance_path)
                    with open(feature_importance_path, 'rb') as img_file:
                        ModelGraph.objects.create(
                            trained_model=ml_model,
                            title="Feature Importance",
                            description="Shows the importance of each feature in the Random Forest model",
                            graph_image=ImageFile(img_file, name=os.path.basename(feature_importance_path))
                        )
                except Exception as e:
                    logger.warning(f"Error generating feature importance graph: {str(e)}")

            except Exception as e:
                logger.warning(f"Error generating graphs: {str(e)}")

            rf_params = {
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'min_samples_split': model.min_samples_split,
                'min_samples_leaf': model.min_samples_leaf,
                'max_features': model.max_features,
                'feature_importances': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None,
                'feature_names': X.columns.tolist()
            }

            return Response({
                "message": "Random Forest model, statistics, and graphs saved successfully.",
                "model": {
                    "id": str(ml_model.id),
                    "name": ml_model.model_name,
                    "type": ml_model.model_type,
                    "target_column": ml_model.target_column,
                    "features": ml_model.features,
                    "is_public": ml_model.is_public,
                    "likes": ml_model.likes,
                    "created_at": ml_model.created_at,
                },
                "metrics": ModelStatsSerializer(ml_model.stats).data if hasattr(ml_model, "stats") else {},
                "graphs": ModelGraphSerializer(ml_model.graphs.all(), many=True).data,
                "model_parameters": rf_params,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error("Unexpected error in RandomForestView.post", exc_info=True)
            return Response({
                "error": "Unexpected server error occurred.",
                "code": "UNEXPECTED_ERROR",
                "details": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)