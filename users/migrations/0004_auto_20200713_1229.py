# Generated by Django 2.0.13 on 2020-07-13 06:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0003_auto_20200713_1224'),
    ]

    operations = [
        migrations.AlterField(
            model_name='flightdatamodel',
            name='WEATHER_DELAY',
            field=models.CharField(default='0', max_length=20),
        ),
    ]
