

from __init__ import *


class DriveServer:

	def __init__(self):
		pass


sio = socketio.Server()
app = Flask(__name__)

speed_limit = 20

def send_control(steering_angle, throttle):
	sio.emit('steer', data ={'steering_angle':steering_angle.__str__(), 'throttle':throttle.__str__()})

@sio.on('telmetry')
def telemetry(sid, data):
	image = Image.open(BytesIO(base64.b64decode(data['image'])))
	image = np.asarray(image)
	image = np.array([image])
	speed = float(data['speed'])
	throttle = 1.0 - speed/speed_limit
	steering_angle = float(model.predict(image))
	send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
	print("connected")
	send_control(0,0)


if __name__ == '__main__':
	model == load_model('/home/ciaran/SmartTech/SmartTechCA2/SmartTechCA2/alpha_model.h5')
	app = socketio.Middleware(sio, app)
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
