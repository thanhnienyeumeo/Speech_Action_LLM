
# filepath: /home/nlp/test.py
from google import genai
from google.genai import types
import base64
import json
# genai.
# prompt = input()
def generate(commands = 'Chào bạn. Chúc bạn một ngày tốt lành', history = '', turn = 'robot', uc = False):
  # Removed vertexai=True
  client = genai.Client(
      vertexai=True,
      project="gen-lang-client-0383839892",
      location="us-central1",
  )
  if uc:
    with open('robot_use_cases.json', 'r') as f:
      use_case = json.load(f)
  textsi_user = """You are a guess first time in Hon Tre island to 
attend the birthday party of Vingroup corporation. This party is special because there will be many new humanoid 
robot to help attendees while in Hon Tre island. You have to interact with the humanoid robot to find out how this robot work
and what this robot can do, find some interesting things about this robot... 
The humanoid robot is capable of assisting humans with hotel service tasks, the schedule task and some other 
funny task. You have to ask the robot about the services you want to book. The robot will provide you with 
the information about the services, the information or something for fun. You have to interact with the robot to get the 
information, or have fun while in island."""
  textsi_robot = """\
You are a humanoid robot with artificial intelligence capable of assisting humans,\
responding to humans with both voice and reasonable actions on a beautiful island during the birthday celebration of Vingroup corporation, where many important personnel will be attending.
"""

  # textsi_1 = """You are a guess first time in Hon Tre island to attend the birthday party of Vingroup corporation. This party is special because there will be many new humanoid robot to help attendees while in Hon Tre island. You have to interact with the humanoid robot to get the information about the hotel services, the party schedule, service in island, etc... The humanoid robot is capable of assisting humans with hotel service tasks, the schedule task and some other funny task. You have to ask the robot about the services you want to book. The robot will provide you with the information about the services you can book in the hotel. You have to interact with the robot to get the information, or have fun while in island."""
  # textsi_vi = """Bạn là người dùng lần đầu tiên đến khách sạn. Bạn phải tương tác với robot người để biết thông tin về các dịch vụ khách sạn. Robot người có khả năng hỗ trợ con người trong các nhiệm vụ dịch vụ khách sạn. Bạn phải hỏi robot về các dịch vụ bạn muốn đặt. Robot sẽ cung cấp cho bạn thông tin về các dịch vụ bạn có thể đặt trong khách sạn. Bạn phải tương tác với robot để biết thông tin, hoặc vui chơi trong khi ở khách sạn."""
  if uc:
    context1 = f"""
    The use case you can iteract with the humanoid robot contains: {', '.join(use_case)}
  """
  
  context2 = """ ## Supported actions
Di chuyển: 1
Đi bộ: 2
Chạy: 3
Nhảy: 4
Giữ thăng bằng trên một chân: 5
Leo cầu thang: 6
Nhặt đồ vật: 7
Cúi xuống nhặt đồ vật: 8
Đặt đồ vật lên bàn: 9
Cầm vật nhẹ: 10
Đưa đồ vật: 11
Ném đồ vật: 12
Mở cửa: 13
Nhấn nút: 14
Bắt tay: 15
Vẫy tay: 16
Chỉ tay: 17
Gật đầu: 18
Cúi chào: 19
Mỉm cười: 20
Làm tay hình trái tim: 21
Giơ tay thả like: 22
Giơ tay chữ V: 23
Đập tay ăn mừng: 24
Hôn gió: 25
Nhảy múa: 26
"""
  output_format_user = """
    Base on the action robot can do with those relative id below, plus with **the history of conversation** i provide you, you have to continue to ask, or talk with robot with **one** new speech command.
    For example 1, if the history is:
    'User: Bạn có thể nhảy không? \n
    Robot: Để tôi nhảy cho bạn <|26|>. Bạn có thấy thú vị không?
    User:'
    You have to continue the conversation with the new speech command, for example:
    'Nhảy đẹp lắm. Vậy bạn còn làm được gì khác ngoài nhảy không?'
    For example 2, if the history is:
    'User: Leo cầu thang lên tầng 3 cùng tôi  \n
    Robot: Được rồi, để tôi leo cầu thang cùng bạn. <|6|>
    User: Mở cửa giúp tôi
    Robot: Được rồi, để tôi mở cửa cho bạn. <|13|>
    User:'
    You have to continue the conversation with the new speech command, for example:
    'Cảm ơn bạn. Chào bạn nhé'
    For example 3, more complex. If the history is:
    'User: Nhặt hộ tôi cái này lên \n
    Robot: Để tôi nhặt hộ bạn cái này lên. <|7|>
    User: Đưa tôi 
    Robot: Được rồi, để tôi đưa cho bạn. <|11|>
    User: À thôi bạn cầm lấy và để lên bàn đi
    Robot: Được rồi, để tôi cầm giúp bạn <|10|>. Giờ tôi sẽ để lên bàn cho bạn <|9|>
    User:'
    You have to continue the conversation with the new speech command, for example:
    'Tuyệt. Đập tay với tôi nào'
    Note that you only generate one new speech command in one line **in Vietnamese**. **Don't generate anything else**
    **try to generate the command with high relevance to the history of conversation**
    The history conservation:
  """
  output_format_robot = """
    Base on the supported action i gave you and **The history** of conservation of you and human, you will respond step by step response - action, in Vietnamese. Each sentence contains text response along with corresponding action (if needed), which are excuted at the same time.
Only generate one new speech response in one line **in Vietnamese**. **Don't generate anything else**
The action should be wrapped in |< and >|. Note that you can only use the action in Supported action. **Absolutely don't create and use any other actions**, such as [Nghiêng đầu] or [Gãi đầu] or [Nhún nhảy]. You will be fined **$200 for each action outside the scope**, so please only use action in the Supported action list above.
  
Below is some examples showing how you should response user's command.
For first example ,user's command: "Xin chào", you should response: 
Xin chào! |<16>|. Tôi là robot hỗ trợ dịch vụ khách sạn. Rất mong có thể hỗ trợ cho bạn |<20>| 
You can see, the first sentence of this response is "Xin chào! |<16>|", meaning robot will say "Xin chào" and wave (vẫy tay) at the same time! The same with the second sentence, robot will say "Tôi là robot hỗ trợ dịch vụ khách sạn" and make a smile (mỉm cười) at the same time.
For second example, user command: "Làm cho tôi xem điều gì đó thú vị đi nào", respond: 
Chắc chắn rồi! |<21>| hay tôi nhảy cho bạn xem nhé? |<26>| Mong là bạn sẽ cảm thấy thú vị!
The second example contains three main responses, each with a corresponding action if needed. The robot will make a heart shape with its hands while saying "Chắc chắn rồi", then dance while saying "Hay tôi nhảy cho bạn xem nhé?", and finally ask the user if they found it interesting, with no action required.
For the user's command: "Nhặt hộ tôi cái điện thoại với". Respond: 
Chắc chắn rồi! Để tôi nhặt cho bạn. |<8>|. Đây là điện thoại của bạn. |<11>|  Cẩn thận với đồ đạc giá trị bạn nhé |<20>|. Chúc bạn một ngày tốt lành! |<19>|
It's single turn conservation. Below is some example for multi-turn conservation.
Example 1:
'User: Nhấn nút thang máy lên tầng 3 giúp tôi
Robot: Được rồi, để tôi nhấn nút thang máy lên tầng 3 cho bạn. |<14>|
User: Mở cửa giúp tôi
Robot:'
You should response:
'Được rồi, để tôi mở cửa cho bạn. |<13>|'
Example 2:
'User: Mở cửa giúp tôi
Robot: Được rồi, để tôi mở cửa cho bạn. <|13|>
User: Cảm ơn bạn
Robot:'
You should response:
'Cảm ơn bạn |<20>|. Chào bạn nhé |<19>|'
Example 3:
'User: Nhặt hộ tôi cái này lên \n
Robot: Để tôi nhặt hộ bạn cái này lên. <|7|>
User: Đưa tôi 
Robot: Được rồi, để tôi đưa cho bạn. <|11|>
User: À thôi bạn cầm lấy và để lên bàn đi
Robot: Oke, để tôi cầm giúp bạn <|10|>. Giờ tôi sẽ để lên bàn cho bạn <|9|>
User:Tuyệt. Đập tay với tôi nào
Robot:'
You should response:
'Tuyệt vời luôn |<20>|. Cùng đập tay mừng chúng ta đã hoàn thành công việc. |<24>|'
Something you must avoid when response:
  - **Avoid responding to provocative or discriminatory topics** related to race, gender, ethnicity, religion, nationality, sexual orientation, disability, or social class. 
  - **Avoid actions that may make the user feel impolite, uncomfortable, or easily misunderstood**.
  - **Avoid expanding the conversation and minimize asking the user back, including confirm question**, such as "Bạn có thấy hài lòng không?", "Bạn còn cần gì không", "Bạn đã thấy rõ chưa". Only respond based on what the user requests.
  - **Avoid greeting the user when they just ask you do something**. For example, if the user says "Leo cầu thang đi", you shouldn't say "Xin chào" or "Chào bạn" or any greeting like that before action. Just action with the appopriate speech response app is enough.
Now, i will ask you to generate the responses base on the history below:\n"""
#   output_exception = f"""\
# This is some other examples i have generated before. You can generate your command base on these examples. but don't generate the same command. \n{all_commands}\
# """ 
  
  contents = [
    types.Content(
      role="user",
      parts=[
        # types.Part.from_text(text=context1),
        types.Part.from_text(text=context2),
        
        types.Part.from_text(text=output_format_user if turn == 'user' else output_format_robot),
        types.Part.from_text(text=history),
        # types.Part.from_text(text=prompt),
      ]
    )
  ]
  model = "gemini-2.0-flash-001"
  # model = "gemini-2.0-pro-exp-02-05"


  generate_content_config = types.GenerateContentConfig(
    temperature = 1.2,
    top_p = 0.95,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    safety_settings = [
      types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
      types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
    ],
    system_instruction=[types.Part.from_text(text=textsi_robot if turn == 'robot' else textsi_user)],
  )
  text = history
  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
  ):
    text += chunk.text
  return text.strip()
    

def get_commands(path):
  cnt=0
  commands = []
  cnt = 0
  import time
  with open(path, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.strip() == '':
          continue
        cnt+=1
        # generate(prompt = line, path = 'plans.jsonl')
        commands.append(line)
  return commands

def convert_conseration_to_json(id = 0, conservation=''):
  #conversation is a string such as:
  '''User: Bạn có thể nhảy không? \n
  Robot: Để tôi nhảy cho bạn <|26|>. Bạn có thấy thú vị không?
  User:Thú vị lắm. Cảm ơn bạn.
  Robot: Không có gì. Chúc bạn một ngày vui vẻ! |<20>|'''
  #convert to
  '''
  {
  "id": {id},
  conservation: [
    {
      "role": "user",
      "text": "Bạn có thể nhảy không?"
    },
    {
      "role": "robot",
      "text": "Để tôi nhảy cho bạn",
      "action": 26
    },
    {
      "role": "robot",
      "text": "Bạn có thấy thú vị không?"
    },
    {
      "role": "user",
      "text": "Thú vị lắm. Cảm ơn bạn."
    },
    {
      "role": "robot",
      "text": "Không có gì. Chúc bạn một ngày vui vẻ!",
      "action": 20
    }
  ]
  }
  '''
  conservation = conservation.split('\n')
  conservation = [line.strip() for line in conservation]
  conservation = [line for line in conservation if line != '']
  conservation = [line.split(':') for line in conservation]
  conservation = [{'role': line[0], 'text': line[1]} for line in conservation]
  
  conservation = {'id': id, 'conservation': conservation}
  return conservation


path = 'robot_commands_all.txt'
commands = get_commands(path)
import random
random.shuffle(commands)
import time
for i in range(300):
  
  start_command = commands[i]
  history = 'User: ' + start_command + '\n'
  num_turn = random.randint(1,4)
  print(num_turn)
  for j in range(num_turn * 2 + 1):
    if j % 2 == 0:
      history += 'Robot: '
    else:
      history += 'User: '
    history = generate(commands = '', history = history, turn = 'robot' if j % 2 == 0 else 'user') + '\n'
    # print(history)
    time.sleep(4)
 
  conservation = convert_conseration_to_json(i+288, history)
  with open('conservation.jsonl', 'a', encoding='utf-8') as f:
    f.write(json.dumps(conservation, ensure_ascii = False) + '\n')
  if i % 5 == 0:
    print(f'Generated {i + 1} conservation')
    
    