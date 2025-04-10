# This example requires OpenAI 1.71, Python 3.12
# To install OpenAI:  pip install openai

import os
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

# Your endpoint should look like the following https://YOUR_OPEN_AI_RESOURCE_NAME.openai.azure.com/
# Set up the Azure OpenAI configuration
client = AzureOpenAI(
  azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("API_KEY"),  
  api_version=os.getenv("API_VERSION")
)

# This will correspond to the custom name you chose for your deployment when you deployed a model.
deployment_id="gpt-35-turbo"

# This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION" which can be configured in an .env file or
# they can be hard coded below using the key and region
speech_config = speechsdk.SpeechConfig(os.getenv("SPEECH_KEY"), os.getenv("SPEECH_REGION"))

# Should be the locale for the speaker's language.
speech_config.speech_recognition_language="en-US"
audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

# The language of the voice that responds on behalf of Azure OpenAI.
# Set up response speech
speech_config.speech_synthesis_voice_name='en-US-JennyMultilingualNeural'
audio_output_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)


speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output_config)

# tts sentence end mark
tts_sentence_end = [ ".", "!", "?", ";", "。", "！", "？", "；", "\n" ]

# Prompts Azure OpenAI with a request and synthesizes the response.
def ask_openai(prompt):
    
    # Ask Azure OpenAI in streaming way
    response = client.chat.completions.create(
        stream=True,
        model=deployment_id,
        messages=[
            {"role": "system", "content": "You will generate comments based on the given data."},
            {"role": "user", "content": prompt}
        ]
    )
    
    collected_messages = []
    last_tts_request = None

    # iterate through the stream response stream
    for chunk in response:
        
        if len(chunk.choices) > 0:
            chunk_message = chunk.choices[0].delta.content  # extract the message
            
            if chunk_message is not None:
                collected_messages.append(chunk_message)  # save the message
                
                if chunk_message in tts_sentence_end: # sentence end found
                    text = ''.join(collected_messages).strip() # join the received message together to build a sentence
                    
                    if text != '': # if sentence only have \n or space, we could skip
                        
                        print(f"Speech synthesized to speaker for: {text}")
                        last_tts_request = speech_synthesizer.speak_text_async(text).get()
                        collected_messages.clear()
            

# Continuously listens for speech input to recognize and send as text to Azure OpenAI
def chat_with_open_ai():
    while True:
        
        print("Azure OpenAI is listening. Say 'Stop' or press Ctrl-Z to end the conversation.")
        
        try:
            # Get audio from the microphone and then send it to the TTS service.
            speech_recognition_result = speech_recognizer.recognize_once_async().get()

            # If speech is recognized, send it to Azure OpenAI and listen for the response.
            if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:

                speech_recognition_result_text = speech_recognition_result.text
                
                if speech_recognition_result_text == "Stop.": 
                    speech_recognition_result_text = "Bye."
                    
                print("Recognized speech: {}".format(speech_recognition_result_text))
                ask_openai(speech_recognition_result_text)
                
                if speech_recognition_result_text == "Bye." or speech_recognition_result_text == "Good Bye.": 
                    print("Conversation ended.")
                    break

            elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
                print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
                break
                
            elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = speech_recognition_result.cancellation_details
                print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print("Error details: {}".format(cancellation_details.error_details))
                    
        except EOFError:
            break

# Main

try:
    chat_with_open_ai()
except Exception as err:
    print("Encountered exception: {}".format(err))