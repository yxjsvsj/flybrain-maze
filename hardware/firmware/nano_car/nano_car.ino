/*
 * Serial control for Arduino Nano + TB6612 (PWMA/PWMB pins).
 * Commands:
 * - V/W line protocol: V<linear> W<angular>\n, values are -1.0 to 1.0
 *   Example: V1.00 W0.00 = forward, V0.00 W1.00 = turn left
 * - Single-char control commands: H/N/P/Q/T
 *   H: high speed  N: normal speed  P: actuator on  Q: actuator off  T: toggle
 * If no command is received within COMMAND_TIMEOUT_MS, the car stops.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Motor driver wiring. Change these only when the TB6612 pins are wired differently.
const int PWMA = 5;  // Motor A PWM pin
const int PWMB = 6;  // Motor B PWM pin
const int AIN1 = 11;
const int AIN2 = 4;
const int BIN1 = 7;
const int BIN2 = 3;

// LED/relay outputs controlled by the non-drive commands P/Q/T.
const int LED1 = A2;
const int LED2 = A3;
const int RELAY1 = A4;
const int RELAY2 = A5;

// Active level settings for output modules. Set to false for active-low hardware.
const bool LED_ACTIVE_HIGH = true;
const bool RELAY_ACTIVE_HIGH = true;

// Set to your TB6612 STBY pin if used; otherwise keep -1 (wired to 5V).
const int STBY = -1;

// PWM limits for speed modes. H selects fast mode, N selects normal/slow mode.
const int PWM_SPEED_FAST = 70;   // 0-255, straight drive limit in fast mode
const int TURN_SPEED_FAST = 60;  // 0-255, turn limit in fast mode
const int PWM_SPEED_SLOW = 30;   // 0-255, straight drive limit in normal mode
const int TURN_SPEED_SLOW = 30;  // 0-255, turn limit in normal mode

// Safety stop: if V/W or control commands stop arriving, motor targets go to zero.
const unsigned long COMMAND_TIMEOUT_MS = 400;

// Soft-start / smoothing. This prevents abrupt direction and PWM changes.
const unsigned long RAMP_INTERVAL_MS = 20; // update interval for PWM ramp
const int RAMP_STEP = 5;                   // PWM change per interval

const long SERIAL_BAUD = 115200;
const int LINE_BUF_SIZE = 48;

// Set true if a motor's direction is reversed.
const bool A_INVERTED = false;
const bool B_INVERTED = false;

struct MotorState {
  bool dir;
  int pwm;
  bool targetDir;
  int targetPwm;
};

MotorState motorA = {true, 0, true, 0};
MotorState motorB = {true, 0, true, 0};
bool actuatorsOn = false;
bool fastMode = true;
int drivePwm = PWM_SPEED_FAST;
int turnPwm = TURN_SPEED_FAST;
char lineBuf[LINE_BUF_SIZE];
int lineLen = 0;

void setup() {
  pinMode(AIN1, OUTPUT);
  pinMode(AIN2, OUTPUT);
  pinMode(BIN1, OUTPUT);
  pinMode(BIN2, OUTPUT);
  pinMode(PWMA, OUTPUT);
  pinMode(PWMB, OUTPUT);

  if (STBY >= 0) {
    pinMode(STBY, OUTPUT);
    digitalWrite(STBY, HIGH);
  }

  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(RELAY1, OUTPUT);
  pinMode(RELAY2, OUTPUT);
  setActuators(false);
  setSpeedMode(true);

  Serial.begin(SERIAL_BAUD);
  stopMotors();
}

void loop() {
  static unsigned long lastCmdMs = 0;

  while (Serial.available() > 0) {
    if (handleSerialChar(Serial.read())) {
      lastCmdMs = millis();
    }
  }

  if (lastCmdMs == 0 || (millis() - lastCmdMs) > COMMAND_TIMEOUT_MS) {
    stopMotors();
  }

  updateMotors();
}

bool handleSerialChar(char c) {
  bool lineMode = lineLen > 0 || c == 'V' || c == 'v' || c == 'W' || c == 'w';
  if (lineMode) {
    if (c == '\n' || c == '\r') {
      if (lineLen == 0) {
        return false;
      }
      lineBuf[lineLen] = '\0';
      lineLen = 0;
      return handleDriveLine(lineBuf);
    }
    if (c >= 'a' && c <= 'z') {
      c = c - 'a' + 'A';
    }
    if (lineLen < LINE_BUF_SIZE - 1) {
      lineBuf[lineLen++] = c;
    } else {
      lineLen = 0;
    }
    return false;
  }

  if (c == '\n' || c == '\r') {
    return false;
  }
  if (c >= 'a' && c <= 'z') {
    c = c - 'a' + 'A';
  }
  if (!isControlCommand(c)) {
    return false;
  }
  processControlCommand(c);
  return true;
}

// These are non-drive commands only:
// H/N choose speed mode, P/Q/T control LED/relay outputs.
// Motion is intentionally handled only by V/W lines, not old single-letter directions.
bool isControlCommand(char c) {
  return c == 'H' || c == 'N' || c == 'P' || c == 'Q' || c == 'T';
}

void processControlCommand(char c) {
  switch (c) {
    case 'H':
      setSpeedMode(true);
      break;
    case 'N':
      setSpeedMode(false);
      break;
    case 'P':
      setActuators(true);
      break;
    case 'Q':
      setActuators(false);
      break;
    case 'T':
      setActuators(!actuatorsOn);
      break;
  }
}

bool handleDriveLine(char *line) {
  char *vPos = strchr(line, 'V');
  char *wPos = strchr(line, 'W');
  if (vPos == NULL || wPos == NULL) {
    return false;
  }
  float v = atof(vPos + 1);
  float w = atof(wPos + 1);
  setDrive(v, w);
  return true;
}

void setTargets(bool aForward, bool bForward, int pwmA, int pwmB) {
  pwmA = constrain(pwmA, 0, 255);
  pwmB = constrain(pwmB, 0, 255);
  motorA.targetDir = aForward;
  motorB.targetDir = bForward;
  motorA.targetPwm = pwmA;
  motorB.targetPwm = pwmB;
}

float clampUnit(float value) {
  if (value > 1.0) {
    return 1.0;
  }
  if (value < -1.0) {
    return -1.0;
  }
  return value;
}

void setDrive(float v, float w) {
  v = clampUnit(v);
  w = clampUnit(w);

  // Differential mix: positive v drives forward, positive w turns left.
  float motorAValue = clampUnit(v + w);
  float motorBValue = clampUnit(v - w);
  int pwmLimit = fabs(v) < 0.01 ? turnPwm : drivePwm;

  int pwmA = (int)(fabs(motorAValue) * pwmLimit);
  int pwmB = (int)(fabs(motorBValue) * pwmLimit);
  setTargets(motorAValue >= 0, motorBValue >= 0, pwmA, pwmB);
}

void updateMotorState(MotorState &m) {
  if (m.targetDir != m.dir && m.pwm > 0) {
    m.pwm = max(0, m.pwm - RAMP_STEP);
    if (m.pwm == 0) {
      m.dir = m.targetDir;
    }
    return;
  }
  if (m.targetPwm > m.pwm) {
    m.pwm = min(m.targetPwm, m.pwm + RAMP_STEP);
  } else if (m.targetPwm < m.pwm) {
    m.pwm = max(m.targetPwm, m.pwm - RAMP_STEP);
  }
}

void updateMotors() {
  static unsigned long lastRampMs = 0;
  unsigned long now = millis();
  if ((now - lastRampMs) < RAMP_INTERVAL_MS) {
    return;
  }
  lastRampMs = now;

  updateMotorState(motorA);
  updateMotorState(motorB);

  if (motorA.pwm == 0) {
    digitalWrite(AIN1, LOW);
    digitalWrite(AIN2, LOW);
  } else {
    bool aDir = motorA.dir ^ A_INVERTED;
    digitalWrite(AIN1, aDir ? LOW : HIGH);
    digitalWrite(AIN2, aDir ? HIGH : LOW);
  }

  if (motorB.pwm == 0) {
    digitalWrite(BIN1, LOW);
    digitalWrite(BIN2, LOW);
  } else {
    bool bDir = motorB.dir ^ B_INVERTED;
    digitalWrite(BIN1, bDir ? LOW : HIGH);
    digitalWrite(BIN2, bDir ? HIGH : LOW);
  }

  analogWrite(PWMA, motorA.pwm);
  analogWrite(PWMB, motorB.pwm);
}

void stopMotors() {
  setTargets(motorA.dir, motorB.dir, 0, 0);
}

void setActuators(bool on) {
  actuatorsOn = on;
  digitalWrite(LED1, on == LED_ACTIVE_HIGH ? HIGH : LOW);
  digitalWrite(LED2, on == LED_ACTIVE_HIGH ? HIGH : LOW);
  digitalWrite(RELAY1, on == RELAY_ACTIVE_HIGH ? HIGH : LOW);
  digitalWrite(RELAY2, on == RELAY_ACTIVE_HIGH ? HIGH : LOW);
}

void setSpeedMode(bool fast) {
  fastMode = fast;
  drivePwm = fast ? PWM_SPEED_FAST : PWM_SPEED_SLOW;
  turnPwm = fast ? TURN_SPEED_FAST : TURN_SPEED_SLOW;
}
