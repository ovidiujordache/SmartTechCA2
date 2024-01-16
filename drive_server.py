

from __init__ import *


class DriveServer:

	def __init__(self):
		pass


sio = socketio.Server()
app = Flask(__name__)

speed_limit = 30

def send_control(steering_angle, throttle):
	print("send control")
	sio.emit('steer', data ={'steering_angle':steering_angle.__str__(), 'throttle':throttle.__str__()})

@sio.on('telemetry')
def telemetry(sid, data):
	image = Image.open(BytesIO(base64.b64decode(data['image'])))
	print("image tele")
	print(image)
	image = np.asarray(image)
	image = np.array([img_preprocess(image)])  # Apply image preprocessing
	speed = float(data['speed'])
	throttle = 1.0 - speed / speed_limit
	steering_angle = float(model.predict(image)[0])  # Ensure you extract the steering angle correctly
	send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
	print("connected")
	# send_control(0,0)

def img_preprocess(img):

  img = img[60:135, :, :]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = cv2.resize(img, (200, 66))
  img = img/255
  return img


if __name__ == '__main__':
	model = load_model('./model/model_1_threelap.h5')
	app = socketio.Middleware(sio, app)
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
