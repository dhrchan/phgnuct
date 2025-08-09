from unified_planning.shortcuts import *
from unified_planning.model.phgn import *
from unified_planning.model.phgn.goal_network import PartialOrderGoalNetwork


def satellite(problem_instance: int = 0):
    problem = PHGNProblem("satellite2")

    # 1. Define UserTypes (with hierarchy)
    # PDDL types: calib_direction, image_direction, instrument, satellite, mode
    # Hierarchy: calib_direction - direction, image_direction - direction
    Object = UserType("Object")
    Direction = UserType("Direction", father=Object)
    CalibDirection = UserType("CalibDirection", father=Direction)
    ImageDirection = UserType("ImageDirection", father=Direction)
    Instrument = UserType("Instrument", father=Object)
    Satellite = UserType("Satellite", father=Object)
    Mode = UserType("Mode", father=Object)

    # 2. Define Fluents (Predicates)
    # All PDDL predicates are boolean in UP
    on_board = problem.add_fluent(
        "on_board", BoolType(), default_initial_value=False, i=Instrument, s=Satellite
    )
    supports = problem.add_fluent(
        "supports", BoolType(), default_initial_value=False, i=Instrument, m=Mode
    )
    pointing = problem.add_fluent(
        "pointing", BoolType(), default_initial_value=False, s=Satellite, d=Direction
    )
    power_avail = problem.add_fluent(
        "power_avail", BoolType(), default_initial_value=False, s=Satellite
    )
    power_on = problem.add_fluent(
        "power_on", BoolType(), default_initial_value=False, i=Instrument
    )
    calibrated = problem.add_fluent(
        "calibrated", BoolType(), default_initial_value=False, i=Instrument
    )
    have_image = problem.add_fluent(
        "have_image", BoolType(), default_initial_value=False, d=ImageDirection, m=Mode
    )
    calibration_target = problem.add_fluent(
        "calibration_target",
        BoolType(),
        default_initial_value=False,
        i=Instrument,
        cd=CalibDirection,
    )
    is_moving = problem.add_fluent(
        "is_moving", BoolType(), default_initial_value=False, d=ImageDirection
    )
    is_stationary = problem.add_fluent(
        "is_stationary", BoolType(), default_initial_value=False, d=ImageDirection
    )
    resolved_motion = problem.add_fluent(
        "resolved_motion", BoolType(), default_initial_value=False, d=ImageDirection
    )

    # 3. Define Actions

    # Action: turn_to
    # :parameters (?t_s - satellite ?t_d_new - direction ?t_d_prev - direction)
    # :precondition (pointing ?t_s ?t_d_prev)
    # :effect (and (pointing ?t_s ?t_d_new) (not (pointing ?t_s ?t_d_prev)))
    turn_to = InstantaneousAction(
        "turn_to", t_s=Satellite, t_d_new=Direction, t_d_prev=Direction
    )
    turn_to.add_precondition(pointing(turn_to.t_s, turn_to.t_d_prev))
    turn_to.add_effect(pointing(turn_to.t_s, turn_to.t_d_new), True)
    turn_to.add_effect(pointing(turn_to.t_s, turn_to.t_d_prev), False)
    problem.add_action(turn_to)

    # Action: detect_motion (Probabilistic due to oneof)
    # :parameters (?t_s - satellite ?t_d - image_direction)
    # :precondition (pointing ?t_s ?t_d)
    # :effect (oneof (is_moving ?t_d) (is_stationary ?t_d))
    detect_motion = ProbabilisticAction(
        "detect_motion", t_s=Satellite, t_d=ImageDirection
    )
    detect_motion.add_precondition(pointing(detect_motion.t_s, detect_motion.t_d))
    # Outcome 1: is_moving
    detect_motion.add_outcome("is_moving_outcome", 0.5)
    detect_motion.add_effect("is_moving_outcome", is_moving(detect_motion.t_d), True)
    # Outcome 2: is_stationary
    detect_motion.add_outcome("is_stationary_outcome", 0.5)
    detect_motion.add_effect(
        "is_stationary_outcome", is_stationary(detect_motion.t_d), True
    )
    problem.add_action(detect_motion)

    # Action: calculate_trajectory
    # :parameters (?t_s - satellite ?t_d - direction)
    # :precondition (and (pointing ?t_s ?t_d) (is_moving ?t_d))
    # :effect (resolved_motion ?t_d)
    calculate_trajectory = InstantaneousAction(
        "calculate_trajectory", t_s=Satellite, t_d=ImageDirection
    )
    calculate_trajectory.add_precondition(
        pointing(calculate_trajectory.t_s, calculate_trajectory.t_d)
    )
    calculate_trajectory.add_precondition(is_moving(calculate_trajectory.t_d))
    calculate_trajectory.add_effect(resolved_motion(calculate_trajectory.t_d), True)
    problem.add_action(calculate_trajectory)

    # Action: fix_instrument_direction
    # :parameters (?t_s - satellite ?t_d - direction)
    # :precondition (and (pointing ?t_s ?t_d) (is_stationary ?t_d))
    # :effect (resolved_motion ?t_d)
    fix_instrument_direction = InstantaneousAction(
        "fix_instrument_direction", t_s=Satellite, t_d=ImageDirection
    )
    fix_instrument_direction.add_precondition(
        pointing(fix_instrument_direction.t_s, fix_instrument_direction.t_d)
    )
    fix_instrument_direction.add_precondition(
        is_stationary(fix_instrument_direction.t_d)
    )
    fix_instrument_direction.add_effect(
        resolved_motion(fix_instrument_direction.t_d), True
    )
    problem.add_action(fix_instrument_direction)

    # Action: switch_on
    # :parameters (?so_i - instrument ?so_s - satellite)
    # :precondition (and (on_board ?so_i ?so_s) (power_avail ?so_s))
    # :effect (and (power_on ?so_i) (not (calibrated ?so_i)) (not (power_avail ?so_s)))
    switch_on = InstantaneousAction("switch_on", so_i=Instrument, so_s=Satellite)
    switch_on.add_precondition(on_board(switch_on.so_i, switch_on.so_s))
    switch_on.add_precondition(power_avail(switch_on.so_s))
    switch_on.add_effect(power_on(switch_on.so_i), True)
    switch_on.add_effect(calibrated(switch_on.so_i), False)
    switch_on.add_effect(power_avail(switch_on.so_s), False)
    problem.add_action(switch_on)

    # Action: switch_off
    # :parameters (?sof_i - instrument ?sof_s - satellite)
    # :precondition (and (on_board ?sof_i ?sof_s) (power_on ?sof_i))
    # :effect (and (not (power_on ?sof_i)) (power_avail ?sof_s))
    switch_off = InstantaneousAction("switch_off", sof_i=Instrument, sof_s=Satellite)
    switch_off.add_precondition(on_board(switch_off.sof_i, switch_off.sof_s))
    switch_off.add_precondition(power_on(switch_off.sof_i))
    switch_off.add_effect(power_on(switch_off.sof_i), False)
    switch_off.add_effect(power_avail(switch_off.sof_s), True)
    problem.add_action(switch_off)

    # Action: calibrate
    # :parameters (?c_s - satellite ?c_i - instrument ?c_d - calib_direction)
    # :precondition (and (on_board ?c_i ?c_s) (calibration_target ?c_i ?c_d) (pointing ?c_s ?c_d) (power_on ?c_i))
    # :effect (calibrated ?c_i)
    calibrate = InstantaneousAction(
        "calibrate", c_s=Satellite, c_i=Instrument, c_d=CalibDirection
    )
    calibrate.add_precondition(on_board(calibrate.c_i, calibrate.c_s))
    calibrate.add_precondition(calibration_target(calibrate.c_i, calibrate.c_d))
    calibrate.add_precondition(pointing(calibrate.c_s, calibrate.c_d))
    calibrate.add_precondition(power_on(calibrate.c_i))
    calibrate.add_effect(calibrated(calibrate.c_i), True)
    problem.add_action(calibrate)

    # Action: take_image
    # :parameters (?ti_s - satellite ?ti_d - image_direction ?ti_i - instrument ?ti_m - mode)
    # :precondition (and (calibrated ?ti_i) (pointing ?ti_s ?ti_d) (on_board ?ti_i ?ti_s) (power_on ?ti_i) (supports ?ti_i ?ti_m))
    # :effect (have_image ?ti_d ?ti_m)
    take_image = InstantaneousAction(
        "take_image", ti_s=Satellite, ti_d=ImageDirection, ti_i=Instrument, ti_m=Mode
    )
    take_image.add_precondition(calibrated(take_image.ti_i))
    take_image.add_precondition(pointing(take_image.ti_s, take_image.ti_d))
    take_image.add_precondition(on_board(take_image.ti_i, take_image.ti_s))
    take_image.add_precondition(power_on(take_image.ti_i))
    take_image.add_precondition(supports(take_image.ti_i, take_image.ti_m))
    take_image.add_effect(have_image(take_image.ti_d, take_image.ti_m), True)
    problem.add_action(take_image)

    #     (:method method0
    # 	:parameters (?mdoatt_t_d_prev - direction ?mdoatt_t_s - satellite ?mdoatt_ti_d - image_direction ?mdoatt_ti_i - instrument ?mdoatt_ti_m - mode)
    # 	:subgoals (and
    # 		(g0 (power_on ?mdoatt_ti_i))
    # 		(g1 (pointing ?mdoatt_t_s ?mdoatt_ti_d))
    # 		(g2 (have_image ?mdoatt_ti_d ?mdoatt_ti_m))
    # 	)
    # 	:precondition (and
    # 		(not (= ?mdoatt_ti_d ?mdoatt_t_d_prev))
    # 	)
    # )
    method0 = PHGNMethod(
        "method0",
        mdoatt_t_d_prev=Direction,
        mdoatt_t_s=Satellite,
        mdoatt_ti_d=ImageDirection,
        mdoatt_ti_i=Instrument,
        mdoatt_ti_m=Mode,
    )
    gn = PartialOrderGoalNetwork()
    gn.add(
        power_on(method0.mdoatt_ti_i),
        pointing(method0.mdoatt_t_s, method0.mdoatt_ti_d),
        have_image(method0.mdoatt_ti_d, method0.mdoatt_ti_m),
    )
    method0.set_goal_network(gn)
    method0.add_precondition(
        Or(
            Not(power_on(method0.mdoatt_ti_i)),
            Not(pointing(method0.mdoatt_t_s, method0.mdoatt_ti_d)),
        )
    )
    problem.add_method(method0)

    # Method 1
    # PDDL:
    # (:method method1
    # 	:parameters (?mdott_t_d_prev - direction ?mdott_t_s - satellite ?mdott_ti_d - image_direction ?mdott_ti_m - mode)
    # 	:subgoals (and (g0 (pointing ?mdott_t_s ?mdott_ti_d)) (g1 (have_image ?mdott_ti_d ?mdott_ti_m)))
    # 	:precondition (and (not (= ?mdott_ti_d ?mdott_t_d_prev)))
    # )
    # method1 = PHGNMethod(
    #     "method1",
    #     mdott_t_d_prev=Direction,
    #     mdott_t_s=Satellite,
    #     mdott_ti_d=ImageDirection,
    #     mdott_ti_m=Mode,
    # )
    # gn1 = PartialOrderGoalNetwork()
    # gn1.add(
    #     pointing(method1.mdott_t_s, method1.mdott_ti_d),
    #     have_image(method1.mdott_ti_d, method1.mdott_ti_m),
    # )
    # method1.set_goal_network(gn1)
    # method1.add_precondition(Not(Equals(method1.mdott_ti_d, method1.mdott_t_d_prev)))
    # problem.add_method(method1)

    # Method 2
    # PDDL:
    # (:method method2
    # 	:parameters (?mdoat_ti_d - image_direction ?mdoat_ti_i - instrument ?mdoat_ti_m - mode)
    # 	:subgoals (and (g0 (power_on ?mdoat_ti_i)) (g1 (have_image ?mdoat_ti_d ?mdoat_ti_m)))
    # )
    # method2 = PHGNMethod(
    #     "method2",
    #     mdoat_ti_d=ImageDirection,
    #     mdoat_ti_i=Instrument,
    #     mdoat_ti_m=Mode,
    # )
    # gn2 = PartialOrderGoalNetwork()
    # gn2.add(
    #     power_on(method2.mdoat_ti_i),
    #     have_image(method2.mdoat_ti_d, method2.mdoat_ti_m),
    # )
    # method2.set_goal_network(gn2)
    # # No preconditions for method2
    # problem.add_method(method2)

    # Method 4
    # PDDL:
    # (:method method4
    # 	:parameters (?maissa_ac_i - instrument ?maissa_sof_i - instrument)
    # 	:subgoals (and (g0 (not (power_on ?maissa_sof_i))) (g1 (power_on ?maissa_ac_i)) (g2 (calibrated ?maissa_ac_i)))
    # 	:precondition (and (not (= ?maissa_sof_i ?maissa_ac_i)))
    # )
    method4 = PHGNMethod(
        "method4",
        maissa_ac_i=Instrument,
        maissa_sof_i=Instrument,
    )
    gn4 = PartialOrderGoalNetwork()
    gn4.add(
        Not(power_on(method4.maissa_sof_i)),  # A negative literal as a subgoal
        power_on(method4.maissa_ac_i),
        calibrated(method4.maissa_ac_i),
    )
    method4.set_goal_network(gn4)
    method4.add_precondition(Not(Equals(method4.maissa_sof_i, method4.maissa_ac_i)))
    problem.add_method(method4)

    # Method 5
    # PDDL:
    # (:method method5
    # 	:parameters (?maisa_ac_i - instrument)
    # 	:subgoals (and (g0 (power_on ?maisa_ac_i)) (g1 (calibrated ?maisa_ac_i)))
    # )
    # method5 = PHGNMethod(
    #     "method5",
    #     maisa_ac_i=Instrument,
    # )
    # gn5 = PartialOrderGoalNetwork()
    # gn5.add(
    #     power_on(method5.maisa_ac_i),
    #     calibrated(method5.maisa_ac_i),
    # )
    # method5.set_goal_network(gn5)
    # # No preconditions for method5
    # problem.add_method(method5)

    # Method 6
    # PDDL:
    # (:method method6
    # 	:parameters (?mactc_c_d - calib_direction ?mactc_c_i - instrument ?mactc_c_s - satellite ?mactc_tt_d_prev - direction)
    # 	:subgoals (and (g0 (pointing ?mactc_c_s ?mactc_c_d)) (g1 (calibrated ?mactc_c_i)))
    # 	:precondition (and (not (= ?mactc_c_d ?mactc_tt_d_prev)))
    # )
    # method6 = PHGNMethod(
    #     "method6",
    #     mactc_c_d=CalibDirection,
    #     mactc_c_i=Instrument,
    #     mactc_c_s=Satellite,
    #     mactc_tt_d_prev=Direction,
    # )
    # gn6 = PartialOrderGoalNetwork()
    # gn6.add(
    #     pointing(method6.mactc_c_s, method6.mactc_c_d),
    #     calibrated(method6.mactc_c_i),
    # )
    # method6.set_goal_network(gn6)
    # method6.add_precondition(Not(Equals(method6.mactc_c_d, method6.mactc_tt_d_prev)))
    # problem.add_method(method6)

    # Method 10
    # PDDL:
    # (:method method10
    # 	:parameters (?dir - image_direction ?mode - mode)
    # 	:subgoals (and (g0 (resolved_motion ?dir)) (g1 (have_image ?dir ?mode)))
    # )
    method10 = PHGNMethod(
        "method10",
        dir=ImageDirection,
        mode=Mode,
    )
    gn10 = PartialOrderGoalNetwork()
    gn10.add(resolved_motion(method10.dir), have_image(method10.dir, method10.mode))
    method10.set_goal_network(gn10)
    method10.add_precondition(Not(resolved_motion(method10.dir)))
    problem.add_method(method10)

    if problem_instance == 1:
        # 1. Define Objects (Constants)
        instrument0 = problem.add_object("instrument0", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        thermograph0 = problem.add_object("thermograph0", Mode)
        GroundStation2 = problem.add_object("GroundStation2", CalibDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument0, satellite0), True)
        problem.set_initial_value(supports(instrument0, thermograph0), True)
        problem.set_initial_value(calibration_target(instrument0, GroundStation2), True)
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)

        # All other fluents not explicitly set are assumed to be False by default
        # (as per default_initial_value=False in fluent definitions)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph0) to have_image(Phenomenon4, thermograph0)
        gn.add(have_image(Phenomenon4, thermograph0))
        problem.set_goal_network(gn)

    if problem_instance == 2:
        # 1. Define Objects (Constants)
        instrument0 = problem.add_object("instrument0", Instrument)
        instrument1 = problem.add_object("instrument1", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        satellite1 = problem.add_object("satellite1", Satellite)
        image1 = problem.add_object("image1", Mode)
        star0 = problem.add_object("star0", CalibDirection)
        star5 = problem.add_object("star5", ImageDirection)
        phenomenon1 = problem.add_object("phenomenon1", ImageDirection)
        phenomenon2 = problem.add_object("phenomenon2", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument0, satellite0), True)
        problem.set_initial_value(supports(instrument0, image1), True)
        problem.set_initial_value(calibration_target(instrument0, star0), True)
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, phenomenon1), True)
        problem.set_initial_value(on_board(instrument1, satellite1), True)
        problem.set_initial_value(supports(instrument1, image1), True)
        problem.set_initial_value(calibration_target(instrument1, star0), True)
        problem.set_initial_value(power_avail(satellite1), True)
        problem.set_initial_value(pointing(satellite1, phenomenon2), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(have_image(star5, image1))
        problem.set_goal_network(gn)

    if problem_instance == 3:
        # 1. Define Objects (Constants)
        instrument0 = problem.add_object("instrument0", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        thermograph0 = problem.add_object("thermograph0", Mode)
        GroundStation2 = problem.add_object("GroundStation2", CalibDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument0, satellite0), True)
        problem.set_initial_value(supports(instrument0, thermograph0), True)
        problem.set_initial_value(calibration_target(instrument0, GroundStation2), True)
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph0) and (do_observation Star5 thermograph0)
        gn.add(have_image(Phenomenon4, thermograph0), have_image(Star5, thermograph0))
        problem.set_goal_network(gn)

    if problem_instance == 4:
        # 1. Define Objects (Constants)
        instrument01 = problem.add_object("instrument01", Instrument)
        instrument02 = problem.add_object("instrument02", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        x_ray = problem.add_object("x_ray", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument01, satellite0), True)
        problem.set_initial_value(supports(instrument01, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument01, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument02, satellite0), True)
        problem.set_initial_value(supports(instrument02, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument02, GroundStation0), True
        )
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph) and (do_observation Star5 x_ray)
        gn.add(have_image(Phenomenon4, thermograph), have_image(Star5, x_ray))
        problem.set_goal_network(gn)

    if problem_instance == 5:
        # 1. Define Objects (Constants)
        instrument0 = problem.add_object("instrument0", Instrument)
        instrument1 = problem.add_object("instrument1", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        satellite1 = problem.add_object("satellite1", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        GroundStation1 = problem.add_object("GroundStation1", CalibDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)
        Phenomenon7 = problem.add_object("Phenomenon7", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument0, satellite0), True)
        problem.set_initial_value(supports(instrument0, thermograph), True)
        problem.set_initial_value(calibration_target(instrument0, GroundStation0), True)
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)
        problem.set_initial_value(on_board(instrument1, satellite1), True)
        problem.set_initial_value(supports(instrument1, thermograph), True)
        problem.set_initial_value(calibration_target(instrument1, GroundStation1), True)
        problem.set_initial_value(power_avail(satellite1), True)
        problem.set_initial_value(pointing(satellite1, Phenomenon7), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph) and (do_observation Star5 thermograph)
        gn.add(have_image(Phenomenon4, thermograph), have_image(Star5, thermograph))
        problem.set_goal_network(gn)

    if problem_instance == 6:
        # 1. Define Objects (Constants)
        instrument01 = problem.add_object("instrument01", Instrument)
        instrument02 = problem.add_object("instrument02", Instrument)
        instrument1 = problem.add_object("instrument1", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        satellite1 = problem.add_object("satellite1", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        x_ray = problem.add_object("x_ray", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        GroundStation1 = problem.add_object("GroundStation1", CalibDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)
        Phenomenon7 = problem.add_object("Phenomenon7", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument01, satellite0), True)
        problem.set_initial_value(supports(instrument01, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument01, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument02, satellite0), True)
        problem.set_initial_value(supports(instrument02, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument02, GroundStation0), True
        )
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)
        problem.set_initial_value(on_board(instrument1, satellite1), True)
        problem.set_initial_value(supports(instrument1, thermograph), True)
        problem.set_initial_value(calibration_target(instrument1, GroundStation1), True)
        problem.set_initial_value(power_avail(satellite1), True)
        problem.set_initial_value(pointing(satellite1, Phenomenon7), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph) and (do_observation Star5 x_ray)
        gn.add(have_image(Phenomenon4, thermograph), have_image(Star5, x_ray))
        problem.set_goal_network(gn)

    if problem_instance == 7:
        # 1. Define Objects (Constants)
        instrument0 = problem.add_object("instrument0", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        thermograph0 = problem.add_object("thermograph0", Mode)
        GroundStation2 = problem.add_object("GroundStation2", CalibDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument0, satellite0), True)
        problem.set_initial_value(supports(instrument0, thermograph0), True)
        problem.set_initial_value(calibration_target(instrument0, GroundStation2), True)
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph0), (do_observation Star5 thermograph0),
        # and (do_observation Phenomenon6 thermograph0)
        gn.add(
            have_image(Phenomenon4, thermograph0),
            have_image(Star5, thermograph0),
            have_image(Phenomenon6, thermograph0),
        )
        problem.set_goal_network(gn)

    if problem_instance == 8:
        # 1. Define Objects (Constants)
        instrument01 = problem.add_object("instrument01", Instrument)
        instrument02 = problem.add_object("instrument02", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        x_ray = problem.add_object("x_ray", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        GroundStation1 = problem.add_object(
            "GroundStation1", CalibDirection
        )  # This object is defined in PDDL but not used in init/goal
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)
        Phenomenon7 = problem.add_object(
            "Phenomenon7", ImageDirection
        )  # This object is defined in PDDL but not used in init/goal
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument01, satellite0), True)
        problem.set_initial_value(supports(instrument01, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument01, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument02, satellite0), True)
        problem.set_initial_value(supports(instrument02, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument02, GroundStation0), True
        )
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph), (do_observation Star5 x_ray),
        # and (do_observation Phenomenon6 x_ray)
        gn.add(
            have_image(Phenomenon4, thermograph),
            have_image(Star5, x_ray),
            have_image(Phenomenon6, x_ray),
        )
        problem.set_goal_network(gn)

    if problem_instance == 9:
        # 1. Define Objects (Constants)
        instrument01 = problem.add_object("instrument01", Instrument)
        instrument02 = problem.add_object("instrument02", Instrument)
        instrument03 = problem.add_object("instrument03", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        x_ray = problem.add_object("x_ray", Mode)
        hd_video = problem.add_object("hd_video", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        Phenomenon7 = problem.add_object(
            "Phenomenon7", ImageDirection
        )  # Defined in PDDL, but not used in init/goal
        Star5 = problem.add_object("Star5", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Phenomenon8 = problem.add_object(
            "Phenomenon8", ImageDirection
        )  # Defined in PDDL, but not used in init/goal
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument01, satellite0), True)
        problem.set_initial_value(supports(instrument01, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument01, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument02, satellite0), True)
        problem.set_initial_value(supports(instrument02, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument02, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument03, satellite0), True)
        problem.set_initial_value(supports(instrument03, hd_video), True)
        problem.set_initial_value(
            calibration_target(instrument03, GroundStation0), True
        )
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph), (do_observation Star5 x_ray),
        # and (do_observation Phenomenon6 hd_video)
        gn.add(
            have_image(Phenomenon4, thermograph),
            have_image(Star5, x_ray),
            have_image(Phenomenon6, hd_video),
        )
        problem.set_goal_network(gn)

    if problem_instance == 10:
        # 1. Define Objects (Constants)
        instrument01 = problem.add_object("instrument01", Instrument)
        instrument11 = problem.add_object("instrument11", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        satellite1 = problem.add_object("satellite1", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        GroundStation1 = problem.add_object("GroundStation1", CalibDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)
        Phenomenon7 = problem.add_object("Phenomenon7", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument01, satellite0), True)
        problem.set_initial_value(supports(instrument01, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument01, GroundStation0), True
        )
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)
        problem.set_initial_value(on_board(instrument11, satellite1), True)
        problem.set_initial_value(supports(instrument11, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument11, GroundStation1), True
        )
        problem.set_initial_value(power_avail(satellite1), True)
        problem.set_initial_value(pointing(satellite1, Phenomenon7), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph), (do_observation Star5 thermograph),
        # and (do_observation Phenomenon6 thermograph)
        gn.add(
            have_image(Phenomenon4, thermograph),
            have_image(Star5, thermograph),
            have_image(Phenomenon6, thermograph),
        )
        problem.set_goal_network(gn)

    if problem_instance == 11:
        # 1. Define Objects (Constants)
        instrument01 = problem.add_object("instrument01", Instrument)
        instrument02 = problem.add_object("instrument02", Instrument)
        instrument11 = problem.add_object("instrument11", Instrument)
        instrument12 = problem.add_object("instrument12", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        satellite1 = problem.add_object("satellite1", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        x_ray = problem.add_object("x_ray", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        GroundStation1 = problem.add_object("GroundStation1", CalibDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)
        Phenomenon7 = problem.add_object("Phenomenon7", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument01, satellite0), True)
        problem.set_initial_value(supports(instrument01, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument01, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument02, satellite0), True)
        problem.set_initial_value(supports(instrument02, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument02, GroundStation0), True
        )
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)
        problem.set_initial_value(on_board(instrument11, satellite1), True)
        problem.set_initial_value(supports(instrument11, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument11, GroundStation1), True
        )
        problem.set_initial_value(on_board(instrument12, satellite1), True)
        problem.set_initial_value(supports(instrument12, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument12, GroundStation1), True
        )
        problem.set_initial_value(power_avail(satellite1), True)
        problem.set_initial_value(pointing(satellite1, Phenomenon7), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph), (do_observation Star5 x_ray),
        # and (do_observation Phenomenon6 x_ray)
        gn.add(
            have_image(Phenomenon4, thermograph),
            have_image(Star5, x_ray),
            have_image(Phenomenon6, x_ray),
        )
        problem.set_goal_network(gn)

    if problem_instance == 12:
        # 1. Define Objects (Constants)
        instrument11 = problem.add_object("instrument11", Instrument)
        instrument01 = problem.add_object("instrument01", Instrument)
        instrument12 = problem.add_object("instrument12", Instrument)
        instrument03 = problem.add_object("instrument03", Instrument)
        instrument02 = problem.add_object("instrument02", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        satellite1 = problem.add_object("satellite1", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        x_ray = problem.add_object("x_ray", Mode)
        hd_video = problem.add_object("hd_video", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        GroundStation1 = problem.add_object("GroundStation1", CalibDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)
        Phenomenon7 = problem.add_object("Phenomenon7", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument01, satellite0), True)
        problem.set_initial_value(supports(instrument01, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument01, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument02, satellite0), True)
        problem.set_initial_value(supports(instrument02, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument02, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument03, satellite0), True)
        problem.set_initial_value(supports(instrument03, hd_video), True)
        problem.set_initial_value(
            calibration_target(instrument03, GroundStation0), True
        )
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)
        problem.set_initial_value(on_board(instrument11, satellite1), True)
        problem.set_initial_value(supports(instrument11, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument11, GroundStation1), True
        )
        problem.set_initial_value(on_board(instrument12, satellite1), True)
        problem.set_initial_value(supports(instrument12, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument12, GroundStation1), True
        )
        problem.set_initial_value(power_avail(satellite1), True)
        problem.set_initial_value(pointing(satellite1, Phenomenon7), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph), (do_observation Star5 x_ray),
        # and (do_observation Phenomenon6 hd_video)
        gn.add(
            have_image(Phenomenon4, thermograph),
            have_image(Star5, x_ray),
            have_image(Phenomenon6, hd_video),
        )
        problem.set_goal_network(gn)

    if problem_instance == 13:
        # 1. Define Objects (Constants)
        instrument0 = problem.add_object("instrument0", Instrument)
        instrument1 = problem.add_object("instrument1", Instrument)
        instrument2 = problem.add_object("instrument2", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        satellite1 = problem.add_object("satellite1", Satellite)
        satellite2 = problem.add_object("satellite2", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        GroundStation1 = problem.add_object("GroundStation1", CalibDirection)
        Phenomenon7 = problem.add_object("Phenomenon7", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Phenomenon8 = problem.add_object("Phenomenon8", ImageDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument0, satellite0), True)
        problem.set_initial_value(supports(instrument0, thermograph), True)
        problem.set_initial_value(calibration_target(instrument0, GroundStation0), True)
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)
        problem.set_initial_value(on_board(instrument1, satellite1), True)
        problem.set_initial_value(supports(instrument1, thermograph), True)
        problem.set_initial_value(calibration_target(instrument1, GroundStation1), True)
        problem.set_initial_value(power_avail(satellite1), True)
        problem.set_initial_value(pointing(satellite1, Phenomenon7), True)
        problem.set_initial_value(on_board(instrument2, satellite2), True)
        problem.set_initial_value(supports(instrument2, thermograph), True)
        problem.set_initial_value(calibration_target(instrument2, GroundStation1), True)
        problem.set_initial_value(power_avail(satellite2), True)
        problem.set_initial_value(pointing(satellite2, Phenomenon8), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph), (do_observation Star5 thermograph),
        # and (do_observation Phenomenon6 thermograph)
        gn.add(
            have_image(Phenomenon4, thermograph),
            have_image(Star5, thermograph),
            have_image(Phenomenon6, thermograph),
        )
        problem.set_goal_network(gn)

    if problem_instance == 14:
        # 1. Define Objects (Constants)
        instrument2 = problem.add_object("instrument2", Instrument)
        instrument11 = problem.add_object("instrument11", Instrument)
        instrument01 = problem.add_object("instrument01", Instrument)
        instrument12 = problem.add_object("instrument12", Instrument)
        instrument02 = problem.add_object("instrument02", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        satellite1 = problem.add_object("satellite1", Satellite)
        satellite2 = problem.add_object("satellite2", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        x_ray = problem.add_object("x_ray", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        GroundStation1 = problem.add_object("GroundStation1", CalibDirection)
        Phenomenon7 = problem.add_object("Phenomenon7", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Phenomenon8 = problem.add_object("Phenomenon8", ImageDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument01, satellite0), True)
        problem.set_initial_value(supports(instrument01, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument01, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument02, satellite0), True)
        problem.set_initial_value(supports(instrument02, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument02, GroundStation0), True
        )
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)
        problem.set_initial_value(on_board(instrument11, satellite1), True)
        problem.set_initial_value(supports(instrument11, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument11, GroundStation1), True
        )
        problem.set_initial_value(on_board(instrument12, satellite1), True)
        problem.set_initial_value(supports(instrument12, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument12, GroundStation1), True
        )
        problem.set_initial_value(power_avail(satellite1), True)
        problem.set_initial_value(pointing(satellite1, Phenomenon7), True)
        problem.set_initial_value(on_board(instrument2, satellite2), True)
        problem.set_initial_value(supports(instrument2, thermograph), True)
        problem.set_initial_value(calibration_target(instrument2, GroundStation1), True)
        problem.set_initial_value(power_avail(satellite2), True)
        problem.set_initial_value(pointing(satellite2, Phenomenon8), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph), (do_observation Star5 x_ray),
        # and (do_observation Phenomenon6 x_ray)
        gn.add(
            have_image(Phenomenon4, thermograph),
            have_image(Star5, x_ray),
            have_image(Phenomenon6, x_ray),
        )
        problem.set_goal_network(gn)

    if problem_instance == 15:
        # 1. Define Objects (Constants)
        instrument2 = problem.add_object("instrument2", Instrument)
        instrument11 = problem.add_object("instrument11", Instrument)
        instrument01 = problem.add_object("instrument01", Instrument)
        instrument12 = problem.add_object("instrument12", Instrument)
        instrument03 = problem.add_object("instrument03", Instrument)
        instrument02 = problem.add_object("instrument02", Instrument)
        satellite0 = problem.add_object("satellite0", Satellite)
        satellite1 = problem.add_object("satellite1", Satellite)
        satellite2 = problem.add_object("satellite2", Satellite)
        thermograph = problem.add_object("thermograph", Mode)
        x_ray = problem.add_object("x_ray", Mode)
        hd_video = problem.add_object("hd_video", Mode)
        GroundStation0 = problem.add_object("GroundStation0", CalibDirection)
        GroundStation1 = problem.add_object("GroundStation1", CalibDirection)
        Phenomenon7 = problem.add_object("Phenomenon7", ImageDirection)
        Star5 = problem.add_object("Star5", ImageDirection)
        Phenomenon4 = problem.add_object("Phenomenon4", ImageDirection)
        Phenomenon8 = problem.add_object("Phenomenon8", ImageDirection)
        Phenomenon6 = problem.add_object("Phenomenon6", ImageDirection)

        # 2. Set Initial State
        problem.set_initial_value(on_board(instrument01, satellite0), True)
        problem.set_initial_value(supports(instrument01, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument01, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument02, satellite0), True)
        problem.set_initial_value(supports(instrument02, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument02, GroundStation0), True
        )
        problem.set_initial_value(on_board(instrument03, satellite0), True)
        problem.set_initial_value(supports(instrument03, hd_video), True)
        problem.set_initial_value(
            calibration_target(instrument03, GroundStation0), True
        )
        problem.set_initial_value(power_avail(satellite0), True)
        problem.set_initial_value(pointing(satellite0, Phenomenon6), True)
        problem.set_initial_value(on_board(instrument11, satellite1), True)
        problem.set_initial_value(supports(instrument11, thermograph), True)
        problem.set_initial_value(
            calibration_target(instrument11, GroundStation1), True
        )
        problem.set_initial_value(on_board(instrument12, satellite1), True)
        problem.set_initial_value(supports(instrument12, x_ray), True)
        problem.set_initial_value(
            calibration_target(instrument12, GroundStation1), True
        )
        problem.set_initial_value(power_avail(satellite1), True)
        problem.set_initial_value(pointing(satellite1, Phenomenon7), True)
        problem.set_initial_value(on_board(instrument2, satellite2), True)
        problem.set_initial_value(supports(instrument2, thermograph), True)
        problem.set_initial_value(calibration_target(instrument2, GroundStation1), True)
        problem.set_initial_value(power_avail(satellite2), True)
        problem.set_initial_value(pointing(satellite2, Phenomenon8), True)

        # 3. Define HTN Goal Network
        gn = PartialOrderGoalNetwork()
        # Translate (do_observation Phenomenon4 thermograph), (do_observation Star5 x_ray),
        # and (do_observation Phenomenon6 hd_video)
        gn.add(
            have_image(Phenomenon4, thermograph),
            have_image(Star5, x_ray),
            have_image(Phenomenon6, hd_video),
        )
        problem.set_goal_network(gn)

    return problem
