class PID:
    def __init__(
        self,
        kp,
        ki,
        kd,
        integral_limit=100.0,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # Maximum absolute value the integral term is allowed to grow to.
        # Without this limit, if the robot is stuck for a long time the
        # integral keeps growing and causes a big overshoot when it finally
        # moves. This is called "integral windup".
        self.integral_limit = integral_limit

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        """
        Clear the stored state between maneuvers.

        Call this before starting a new goto() movement so that the
        integral and derivative from the previous move do not carry over.
        """
        self.integral = 0.0
        self.prev_error = 0.0

    def update(
        self,
        error,
        dt,
    ):
        """
        Compute one PID output step.

        Parameters
        ----------
        error : float
            Difference between setpoint and measured value.
            For heading control this is the heading error in degrees.
        dt : float
            Time since the last call in seconds.

        Returns
        -------
        float
            The correction to apply (e.g. speed difference between motors).
        """

        # --- Proportional ---
        P = self.kp * error

        # --- Integral (with windup clamp) ---
        self.integral += error * dt
        self.integral = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral),
        )
        I = self.ki * self.integral

        # --- Derivative ---
        # Guard against dt=0 to avoid division by zero.
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
        D = self.kd * derivative

        output = P + I + D

        self.prev_error = error

        return output