from rest_framework import serializers
from .models import TrainedModel, ModelStats, ModelGraph

from rest_framework import serializers
from .models import TrainedModel, ModelStats, ModelGraph

class ModelStatsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelStats
        fields = [
            'id',
            'trained_model',
            'r2_score',
            'mse',
            'mae',
            'accuracy',
            'precision',
            'recall',
            'f1_score'
        ]

class ModelGraphSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelGraph
        fields = [
            'id',
            'trained_model',
            'title',
            'description',
            'graph_image',
            'graph_json'
        ]
        

class TrainedModelSerializer(serializers.ModelSerializer):
    stats = ModelStatsSerializer(read_only=True)
    graphs = ModelGraphSerializer(many=True, read_only=True)

    class Meta:
        model = TrainedModel
        fields = [
            'id',
            'user_id',
            'model_name',
            'model_type',
            'polynomial_degree',
            'target_column',
            'features',
            'csv_file',
            'model_file',
            'created_at',
            'is_public',
            'likes',
            'stats', 
            'graphs',
        ]
        read_only_fields = ['id', 'created_at', 'stats', 'graphs']