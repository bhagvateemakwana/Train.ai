from django.db import models
import uuid
from accounts.models import User

# Create your models here.
class TrainedModel(models.Model):
    class ModelType(models.TextChoices):
        LINEAR_REGRESSION = 'LinearRegression', 'Linear Regression'
        POLYNOMIAL_REGRESSION = 'PolynomialRegression', 'Polynomial Regression'
        DECISION_TREE = 'DecisionTree', 'Decision Tree'
        KNN = 'KNN', 'K-Nearest Neighbors'
        RANDOM_FOREST = 'RandomForest', 'Random Forest',
        RIDGE_REGRESSION = 'RidgeRegression', 'Ridge Regression'

    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='trained_models')
    model_type = models.CharField(
        max_length=50,
        choices=ModelType.choices
    )
    polynomial_degree = models.IntegerField(null=True, blank=True)
    model_name = models.CharField(max_length=100)
    target_column = models.CharField(max_length=100)
    features = models.TextField(null=True, blank=True)
    model_file = models.FileField(upload_to='models/', null=True, blank=True)
    csv_file = models.FileField(upload_to='data/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    is_public = models.BooleanField(default=False)
    likes = models.IntegerField(default=0)

    def __str__(self):
        return f"{self.model_type} | Target: {self.target_column}"
    
class ModelStats(models.Model):
    trained_model = models.OneToOneField(TrainedModel, on_delete=models.CASCADE, related_name='stats')

    # Regression metrics
    r2_score = models.FloatField(null=True, blank=True)
    mse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)

    # Classification metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"Stats for {self.trained_model.model_name}"

class ModelGraph(models.Model):
    trained_model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE, related_name='graphs')
    title = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    graph_image = models.ImageField(upload_to='graphs/', null=True, blank=True)
    graph_json = models.TextField(null=True, blank=True)

    def __str__(self):
        return f"Graph: {self.title} for {self.trained_model.model_name}"