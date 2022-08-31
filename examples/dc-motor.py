######################################################################################
# Uses Collimtaor to calculate                                                       #
# the optimal values for a DC motor shaft position                                   #
# See here: https://www.collimator.ai/tutorials/dc-motor-position-controller-design  #
# for full tutorial                                                                  #
######################################################################################

import pycollimator as collimator
import matplotlib.pyplot as plt

token = "cbc28a3a-50c4-4903-8396-c186b45b460d"
project_uuid = "09b12fa1-8d06-4a60-9c7f-2b4930df6732"
collimator.set_auth_token(token,project_uuid)

collimator.list_models()
dc_motor_no_controller = collimator.load_model('DC Motor')

dc_motor_no_controller.set_parameters({"J": 100})
dc_motor_no_controller.parameters.b = 0.1
dc_motor_no_controller.parameters.Ke = 0.01
dc_motor_no_controller.parameters.Kt = 0.01
dc_motor_no_controller.parameters.R = 1
dc_motor_no_controller.parameters.L = 2
print("Parameters:", dc_motor_no_controller.parameters)

dc_motor_no_controller.configuration.stop_time = 10
print("Configuration:", dc_motor_no_controller.configuration)

results = collimator.run_simulation(dc_motor_no_controller).results

plt.plot(results['time'],results['Integrator_0'])
plt.xlabel('Time')
plt.ylabel('Shaft Position')
#plt.show()

dc_motor_pi_controller = collimator.load_model('DC Motor PI')
dc_motor_pi_controller.get_parameters()

dc_motor_pi_controller.parameters.Ki = 0.1
dc_motor_pi_controller.parameters.Kp = 50
dc_motor_pi_controller.configuration.stop_time = 10

results_pi = collimator.run_simulation(dc_motor_pi_controller).results

plt.plot(results_pi['time'],results_pi['Outport'])
plt.xlabel('Time')
plt.ylabel('Shaft Position')
#plt.show()

dc_motor_lag_controller = collimator.load_model('DC Motor Lag')
dc_motor_lag_controller.set_parameters({"Kp": 50})
dc_motor_lag_controller.set_configuration({"stop_time": 10})
results_lag = collimator.run_simulation(dc_motor_lag_controller).results


plt.plot(results['time'], results['Outport'], label = 'Motor Response')
plt.plot(results_lag['time'], results_lag['Outport'], label = 'Lag Controller Response')
plt.plot(results_pi['time'], results_pi['Outport'], label = 'PI Controller Response')
plt.plot(results['time'], results['Voltage'], label = 'Ideal Response')
plt.xlabel('Time')
plt.ylabel('Shaft Position')
plt.legend()
plt.show()
