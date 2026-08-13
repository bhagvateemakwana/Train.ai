from django.contrib import admin
from .models import TrainedModel, ModelStats, ModelGraph

# Register your models here.
admin.site.register(TrainedModel)
admin.site.register(ModelStats)
admin.site.register(ModelGraph)