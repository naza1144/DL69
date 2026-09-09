from django.contrib import admin
from .models import MLPModel


@admin.register(MLPModel)
class MLPModelAdmin(admin.ModelAdmin):
    list_display = ("id", "label_rule", "center_x", "center_y", "rotation", "file_path", "create_on", "update_on")
    list_filter = ("label_rule",)
    search_fields = ("file_path",)
    readonly_fields = ("create_on", "update_on")
