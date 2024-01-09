from django import forms
from users.models import FlightDataModel

class FlightDataForms(forms.ModelForm):
    DAY = forms.IntegerField()
    DEPARTURE_TIME = forms.FloatField()
    FLIGHT_NUMBER = forms.IntegerField()
    DESTINATION_AIRPORT = forms.CharField(max_length=100)
    ORIGIN_AIRPORT = forms.CharField(max_length=100)
    DAY_OF_WEEK = forms.IntegerField()
    TAXI_OUT = forms.FloatField()

    class Meta():
        model = FlightDataModel
        fields = '__all__'