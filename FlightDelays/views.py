from django.shortcuts import render
from users.forms import UserRegistrationForm

def index(request):
    return render(request,'index.html',{})

def UserRegister(request):
    form = UserRegistrationForm()
    return render(request,'UserRegister.html',{'form':form})

def UserLogin(request):
    return render(request,'UserLogin.html',{})

def AdminLogin(request):
    return render(request,'AdminLogin.html',{})

def Logout(request):
    return render(request,'index.html',{})