print("Hello World")
Hello World

import random
import requests
def main():
    print("Hello! What's up for the day.")
    second_main()

def second_main():
    try:
        from textblob import TextBlob
        input_type = raw_input("Main Menu: ")
        cleaned = preprocess_text(input_type)
        parsed = TextBlob(cleaned)
        words_in_sentence = parsed.split()
        if ("book" in words_in_sentence.lower()) or ("cab" in words_in_sentence.lower()):
            check_for_Cab_Booking(parsed)
        if ("weather" in parsed.lower()) or ("temperature" in parsed.lower()):
            check_for_Weather(parsed)

        GREETING_KEYWORDS = ("hello", "hi", "greetings", "sup", "what's up","good", "gud", "hey","cheers","good morning",)
        for word in parsed.words:
            if word.lower() in GREETING_KEYWORDS :
                check_for_greetings(parsed)
    except:
        random_responses()
    random_responses()

def preprocess_text(sentence):
    cleaned = []
    words = sentence.split(' ')
    for w in words:
        if w == 'i':
            w = 'I'
        if w == "i'm":
            w = "I'm"
        if w == "wanna" :
            w = "want to"
        if w == "don't" :
            w ="do not"
        cleaned.append(w)

    return ' '.join(cleaned)

def check_for_Weather(parsed):
    try:
        from geotext import GeoText
        places = GeoText(str(parsed))
        City_Name =places.cities[0]
        #City_Name= "Give the name of the city after preprocessing the text"
    except:
        #Validation checker
        weather_validation()
    get_weather(City_Name)

def get_weather(City_Name):
    url = 'http://api.openweathermap.org/data/2.5/weather?q=%s,ua&appid=%s'
    key = '65e35748bfe2f7d0e48946797b374b44'
    
    city = City_Name
    r = requests.get(url % (city, key))
    json_r = r.json()
    temp_kel = json_r['main']['temp']
    precipitation = json_r['weather'][0]['description']
    wind = json_r['wind']['speed']
    temp_cel = round(temp_kel-273.15)
    humidity=json_r['main']['humidity']
    #t = Temp(temp_cel, 'c')
    #hi = heat_index(temperature=t, humidity=humidity)

    print("Current temperature in "+city+' : ' + str(temp_cel))  
    print ("Precipitation: "+precipitation)
    print ("Wind speed: %s meters per second" % wind)
    print ("Humidity: %s" % str(humidity)+'%')
    #print ("Feel\'s like: %s" % str(round((hi.c)))+'C')
    print ('-----------------------------------------------')
    print(' \n')
    print("Enter \"Yes\" for knowing another city weather Update ")
    input_type_w_next = raw_input("Enter your Option : ")
    if (input_type_w_next.lower() == "yes") or (input_type_w_next.lower() == "y") or (input_type_w_next.lower() == "yah"):
        weather_validation()
    else:
        print("Sorry you entered invalid option, So switching back to Main Menu")
        print("-----------------------------------------------------------------")
        second_main()    

def weather_validation(): #Validation checker
    print("Please enter a Location for Weather Updates")
    try:
        second_input_w = raw_input()
        from geotext import GeoText
        from textblob import TextBlob
        parsed = TextBlob(second_input_w)
        places = GeoText(str(parsed))
        City_Name =places.cities[0]
    except:
        print("you didn't entered a valid location again")
        print("type  YES to continue with weather updates and NO to switch back to main menu")
        weather_final_check = raw_input()
        if (weather_final_check == "yes") or (weather_final_check == "y") or (weather_final_check =="yah"):
            weather_validation()
        else:
            second_main()
    get_weather(City_Name)

def check_for_Cab_Booking(parsed):
    try:
        import nltk
        words =",".join(str(x) for x in nltk.word_tokenize(str(parsed)))
        from geotext import GeoText
        places = GeoText(words)
        pickup_location =places.cities[0]
        drop_location = places.cities[1]
    except:
        #print("Please provide both Pick-up and Drop-location")
        cab_validation()
    verified_cab_booking(pickup_location, drop_location)

def cab_validation():
    
    try:
        print("Enter both Pick-up and Drop Locations")
        text_input_c = raw_input(">: ")
        import nltk
        words =" ".join(str(x) for x in nltk.word_tokenize(text_input_c))
        from geotext import GeoText
        places = GeoText(words)
        pickup_location =places.cities[0]
        drop_location = places.cities[1]
    except:
        print("you didn't entered a valid locations again")
        print("type  YES to continue with cab booking services and No to switch back to main menu")
        cab_final_check = raw_input()
        if (cab_final_check.lower() == "yes") or (cab_final_check.lower() == "y") or (cab_final_check.lower() =="yah"):
            cab_validation()
        else:
            second_main()
    verified_cab_booking(pickup_location, drop_location)

def verified_cab_booking(pickup_location, drop_location):
    pickup_location = pickup_location
    drop_location = drop_location
    print("Your Pick-up point is: %s and your dropping point is %s "%(pickup_location,drop_location))
    print("please confirm your details Yes/No., or if you want cancel this booking type X or Cancel")
     
    cab_verify = raw_input(">: ")
    if(cab_verify.lower() == "yes") or (cab_verify.lower() =="y") or (cab_verify.lower() =="yah") :
        payment_details(pickup_location, drop_location)
    elif(cab_verify.lower() == "no") or (cab_verify.lower() =="n"):
        cab_validation()
    elif(cab_verify.lower() == "x") or (cab_verify.lower() =="cancel"):
        second_main()
    else:
        print("Sorry your options are wrong., rolling back to main menu")
        second_main()

def payment_details(pickup_location,drop_location):
    #this is related to payment Gateway
    #So here we Book the cab by default
    #print("Cab is booked from %s to %s",%pickup_location,%drop_location)
    print("Your payment has been processed")
    print("Cab is booked")
    print("Please do visit again")
    print("              __________/\__________              ")
    print("\n")
    second_main()

def check_for_greetings(parsed):
    GREETING_RESPONSES = ["'sup bro","Good Day" ,"hey", "good to know", "hey you get my snap?"]
    print(random.choice(GREETING_RESPONSES))
    second_main()

def random_responses():
    RANDOM_RESPONSES = ["That sounds great","stay healthy" ,"cheers", "Sarcastic", "Are you feeling well"]
    print(random.choice(RANDOM_RESPONSES))
    second_main()

if __name__ == "__main__":
    main()
