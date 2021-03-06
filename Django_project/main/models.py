from django.db import models

class ToDoList(models.Model):
    name = models.CharField(max_length=200)

    def __str__(self):
        return self.name

class Item(models.Model):
    ToDoList = models.ForeignKey(ToDoList, on_delete=models.CASCADE)
    text = models.TextField(max_length=300)
    complete = models.BooleanField()

    def __str__(self):
        return self.text