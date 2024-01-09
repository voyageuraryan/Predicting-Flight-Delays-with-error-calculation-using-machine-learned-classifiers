from django.db import models


# Create your models here.

class UserRegistrationModel(models.Model):
    name = models.CharField(max_length=100)
    loginid = models.CharField(unique=True, max_length=100)
    password = models.CharField(max_length=100)
    mobile = models.CharField(max_length=100)
    email = models.CharField(max_length=100)
    locality = models.CharField(max_length=100)
    address = models.CharField(max_length=1000)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    status = models.CharField(max_length=100)

    def __str__(self):
        return self.loginid

    class Meta:
        db_table = 'AviationUsers'


class FlightDataModel(models.Model):
    DAY =models.IntegerField(default=0)
    DEPARTURE_TIME =models.FloatField(default=0.0)
    FLIGHT_NUMBER =models.IntegerField(default=0)
    DESTINATION_AIRPORT =models.CharField(max_length=100)
    ORIGIN_AIRPORT =models.CharField(max_length=100)
    DAY_OF_WEEK =models.IntegerField(default=0)
    TAXI_OUT =models.FloatField(default=0.0)
    def __str__(self):
        return self.id

    class Meta:
        db_table = "FlighDelayData"
