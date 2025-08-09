from unified_planning.shortcuts import *
from unified_planning.model.phgn import *
from unified_planning.model.phgn.goal_network import PartialOrderGoalNetwork


def transport(problem_instance: int = 1):
    # --- TRANSPORT DOMAIN DEFINITION (from previous response) ---
    problem = PHGNProblem()

    # 1. Define UserTypes (with hierarchy)
    Object = UserType("Object")
    Locatable = UserType("Locatable", father=Object)
    Package = UserType("Package", father=Locatable)
    Vehicle = UserType("Vehicle", father=Locatable)
    CapacityNumber = UserType("CapacityNumber", father=Object)
    Location = UserType("Location", father=Object)
    Target = UserType("Target", father=Object)

    # 2. Define Fluents (Predicates)
    road = problem.add_fluent(
        "road", BoolType(), default_initial_value=False, loc1=Location, loc2=Location
    )
    at = problem.add_fluent(
        "at", BoolType(), default_initial_value=False, obj=Locatable, loc=Location
    )
    in_vehicle = problem.add_fluent(
        "in_vehicle", BoolType(), default_initial_value=False, p=Package, v=Vehicle
    )
    current_capacity_level = problem.add_fluent(
        "current_capacity_level",
        BoolType(),
        default_initial_value=False,
        v=Vehicle,
        s=CapacityNumber,
    )
    capacity_predecessor = problem.add_fluent(
        "capacity_predecessor",
        BoolType(),
        default_initial_value=False,
        s1=CapacityNumber,
        s2=CapacityNumber,
    )

    # 3. Define Actions
    drive = InstantaneousAction("drive", v=Vehicle, l1=Location, l2=Location)
    drive.add_precondition(at(drive.v, drive.l1))
    drive.add_precondition(road(drive.l1, drive.l2))
    drive.add_precondition(Not(Equals(drive.l1, drive.l2)))
    drive.add_effect(at(drive.v, drive.l1), False)
    drive.add_effect(at(drive.v, drive.l2), True)
    problem.add_action(drive)

    pick_up = InstantaneousAction(
        "pick_up",
        v=Vehicle,
        l=Location,
        p=Package,
        s1=CapacityNumber,
        s2=CapacityNumber,
    )
    pick_up.add_precondition(at(pick_up.v, pick_up.l))
    pick_up.add_precondition(at(pick_up.p, pick_up.l))
    pick_up.add_precondition(capacity_predecessor(pick_up.s1, pick_up.s2))
    pick_up.add_precondition(current_capacity_level(pick_up.v, pick_up.s2))
    pick_up.add_effect(at(pick_up.p, pick_up.l), False)
    pick_up.add_effect(in_vehicle(pick_up.p, pick_up.v), True)
    pick_up.add_effect(current_capacity_level(pick_up.v, pick_up.s1), True)
    pick_up.add_effect(current_capacity_level(pick_up.v, pick_up.s2), False)
    problem.add_action(pick_up)

    drop = ProbabilisticAction(
        "drop", v=Vehicle, l=Location, p=Package, s1=CapacityNumber, s2=CapacityNumber
    )
    drop.add_precondition(at(drop.v, drop.l))
    drop.add_precondition(in_vehicle(drop.p, drop.v))
    drop.add_precondition(capacity_predecessor(drop.s1, drop.s2))
    drop.add_precondition(current_capacity_level(drop.v, drop.s1))

    drop.add_outcome("success", 0.5)
    drop.add_effect("success", in_vehicle(drop.p, drop.v), False)
    drop.add_effect("success", at(drop.p, drop.l), True)
    drop.add_effect("success", current_capacity_level(drop.v, drop.s2), True)
    drop.add_effect("success", current_capacity_level(drop.v, drop.s1), False)
    drop.add_outcome("failure", 0.5)
    problem.add_action(drop)

    deliver = PHGNMethod(
        "deliver",
        package=Package,
        vehicle=Vehicle,
        source=Location,
        destination=Location,
    )
    deliver.add_precondition(at(deliver.package, deliver.source))
    deliver.add_precondition(Not(Equals(deliver.source, deliver.destination)))
    gn = PartialOrderGoalNetwork()
    gn.add(
        at(deliver.vehicle, deliver.source),
        in_vehicle(deliver.package, deliver.vehicle),
        And(
            at(deliver.vehicle, deliver.destination),
            in_vehicle(deliver.package, deliver.vehicle),
        ),
        at(deliver.package, deliver.destination),
    )
    deliver.set_goal_network(gn)
    problem.add_method(deliver)

    if problem_instance == 1:
        # --- PROBLEM INSTANCE DEFINITION ---

        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)

        problem.set_initial_value(road(city_loc_0, city_loc_1), True)
        problem.set_initial_value(road(city_loc_1, city_loc_0), True)
        problem.set_initial_value(road(city_loc_1, city_loc_2), True)
        problem.set_initial_value(road(city_loc_2, city_loc_1), True)

        problem.set_initial_value(at(package_0, city_loc_1), True)
        problem.set_initial_value(at(package_1, city_loc_1), True)
        problem.set_initial_value(at(truck_0, city_loc_2), True)

        problem.set_initial_value(current_capacity_level(truck_0, capacity_2), True)

        gn = PartialOrderGoalNetwork()
        gn.add(at(package_0, city_loc_0), at(package_1, city_loc_2))
        problem.set_goal_network(gn)

    if problem_instance == 2:
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)

        problem.set_initial_value(road(city_loc_0, city_loc_3), True)
        problem.set_initial_value(road(city_loc_1, city_loc_2), True)
        problem.set_initial_value(road(city_loc_1, city_loc_3), True)
        problem.set_initial_value(road(city_loc_2, city_loc_1), True)
        problem.set_initial_value(road(city_loc_3, city_loc_0), True)
        problem.set_initial_value(road(city_loc_3, city_loc_1), True)

        problem.set_initial_value(at(package_0, city_loc_3), True)
        problem.set_initial_value(at(package_1, city_loc_2), True)
        problem.set_initial_value(at(package_2, city_loc_2), True)
        problem.set_initial_value(at(truck_0, city_loc_3), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_3), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            at(package_0, city_loc_1),
            at(package_1, city_loc_0),
            at(package_2, city_loc_0),
        )
        problem.set_goal_network(gn)

    if problem_instance == 3:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)

        problem.set_initial_value(road(city_loc_0, city_loc_0), True)
        problem.set_initial_value(road(city_loc_0, city_loc_1), True)
        problem.set_initial_value(road(city_loc_1, city_loc_0), True)
        problem.set_initial_value(road(city_loc_1, city_loc_1), True)
        problem.set_initial_value(road(city_loc_1, city_loc_2), True)
        problem.set_initial_value(road(city_loc_2, city_loc_1), True)
        problem.set_initial_value(road(city_loc_2, city_loc_2), True)

        problem.set_initial_value(at(package_0, city_loc_1), True)
        problem.set_initial_value(at(package_1, city_loc_2), True)
        problem.set_initial_value(at(package_2, city_loc_2), True)

        problem.set_initial_value(at(truck_0, city_loc_0), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_3), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        t1, t0 = gn.add(at(package_1, city_loc_1), at(package_0, city_loc_0))
        t2 = gn.add(at(package_2, city_loc_0))
        gn.add_ordering(t1, t2)
        problem.set_goal_network(gn)

    if problem_instance == 4:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)
        capacity_4 = problem.add_object("capacity_4", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)
        problem.set_initial_value(capacity_predecessor(capacity_3, capacity_4), True)

        problem.set_initial_value(road(city_loc_0, city_loc_3), True)
        problem.set_initial_value(road(city_loc_1, city_loc_2), True)
        problem.set_initial_value(road(city_loc_2, city_loc_1), True)
        problem.set_initial_value(road(city_loc_2, city_loc_3), True)
        problem.set_initial_value(road(city_loc_3, city_loc_0), True)
        problem.set_initial_value(road(city_loc_3, city_loc_2), True)
        problem.set_initial_value(road(city_loc_3, city_loc_3), True)

        problem.set_initial_value(at(package_0, city_loc_0), True)
        problem.set_initial_value(at(package_1, city_loc_1), True)
        problem.set_initial_value(at(package_2, city_loc_3), True)
        problem.set_initial_value(at(package_3, city_loc_2), True)

        problem.set_initial_value(at(truck_0, city_loc_0), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_4), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_3))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_0))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_1))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_0))  # Corresponds to PDDL task3

        # Add ordering constraints
        gn.add_ordering(t0, t2)  # (< task0 task2)
        gn.add_ordering(t2, t1)  # (< task2 task1)
        gn.add_ordering(t2, t3)  # (< task2 task3)

        problem.set_goal_network(gn)

    if problem_instance == 5:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)
        package_4 = problem.add_object("package_4", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)
        capacity_4 = problem.add_object("capacity_4", CapacityNumber)
        capacity_5 = problem.add_object("capacity_5", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)
        problem.set_initial_value(capacity_predecessor(capacity_3, capacity_4), True)
        problem.set_initial_value(capacity_predecessor(capacity_4, capacity_5), True)

        problem.set_initial_value(road(city_loc_0, city_loc_2), True)
        problem.set_initial_value(road(city_loc_1, city_loc_1), True)
        problem.set_initial_value(road(city_loc_1, city_loc_3), True)
        problem.set_initial_value(road(city_loc_2, city_loc_0), True)
        problem.set_initial_value(road(city_loc_2, city_loc_3), True)
        problem.set_initial_value(road(city_loc_3, city_loc_1), True)
        problem.set_initial_value(road(city_loc_3, city_loc_2), True)

        problem.set_initial_value(at(package_0, city_loc_0), True)
        problem.set_initial_value(at(package_1, city_loc_2), True)
        problem.set_initial_value(at(package_2, city_loc_0), True)
        problem.set_initial_value(at(package_3, city_loc_0), True)
        problem.set_initial_value(at(package_4, city_loc_1), True)

        problem.set_initial_value(at(truck_0, city_loc_1), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_5), True)

        # 3. Set Goal Network
        # The 'deliver' tasks in PDDL HTN typically correspond to 'at' goals for the package
        # at the specified location in the Unified Planning library's HTN representation.
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_1))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_3))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_1))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_1))  # Corresponds to PDDL task3
        t4 = gn.add(at(package_4, city_loc_2))  # Corresponds to PDDL task4

        # Add ordering constraints
        gn.add_ordering(t1, t4)  # (< task1 task4)
        gn.add_ordering(t4, t2)  # (< task4 task2)
        gn.add_ordering(t4, t3)  # (< task4 task3)
        gn.add_ordering(t4, t0)  # (< task4 task0)

        problem.set_goal_network(gn)

    if problem_instance == 6:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)
        package_4 = problem.add_object("package_4", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)
        capacity_4 = problem.add_object("capacity_4", CapacityNumber)
        capacity_5 = problem.add_object("capacity_5", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)
        city_loc_4 = problem.add_object("city_loc_4", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)
        problem.set_initial_value(capacity_predecessor(capacity_3, capacity_4), True)
        problem.set_initial_value(capacity_predecessor(capacity_4, capacity_5), True)

        problem.set_initial_value(road(city_loc_0, city_loc_1), True)
        problem.set_initial_value(road(city_loc_1, city_loc_0), True)
        problem.set_initial_value(road(city_loc_1, city_loc_1), True)
        problem.set_initial_value(road(city_loc_1, city_loc_2), True)
        problem.set_initial_value(road(city_loc_1, city_loc_3), True)
        problem.set_initial_value(road(city_loc_2, city_loc_1), True)
        problem.set_initial_value(road(city_loc_2, city_loc_4), True)
        problem.set_initial_value(road(city_loc_3, city_loc_1), True)
        problem.set_initial_value(road(city_loc_3, city_loc_3), True)
        problem.set_initial_value(road(city_loc_4, city_loc_2), True)

        problem.set_initial_value(at(package_0, city_loc_3), True)
        problem.set_initial_value(at(package_1, city_loc_3), True)
        problem.set_initial_value(at(package_2, city_loc_2), True)
        problem.set_initial_value(at(package_3, city_loc_3), True)
        problem.set_initial_value(at(package_4, city_loc_1), True)

        problem.set_initial_value(at(truck_0, city_loc_4), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_5), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_0))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_4))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_4))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_2))  # Corresponds to PDDL task3
        t4 = gn.add(at(package_4, city_loc_2))  # Corresponds to PDDL task4

        # Add ordering constraints
        gn.add_ordering(t0, t3)  # (< task0 task3)
        gn.add_ordering(t0, t4)  # (< task0 task4)
        gn.add_ordering(t4, t2)  # (< task4 task2)
        gn.add_ordering(t4, t1)  # (< task4 task1)
        gn.add_ordering(t3, t2)  # (< task3 task2)
        gn.add_ordering(t3, t1)  # (< task3 task1)

        problem.set_goal_network(gn)

    if problem_instance == 7:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)
        package_4 = problem.add_object("package_4", Package)
        package_5 = problem.add_object("package_5", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)
        capacity_4 = problem.add_object("capacity_4", CapacityNumber)
        capacity_5 = problem.add_object("capacity_5", CapacityNumber)
        capacity_6 = problem.add_object("capacity_6", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)
        city_loc_4 = problem.add_object("city_loc_4", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)
        problem.set_initial_value(capacity_predecessor(capacity_3, capacity_4), True)
        problem.set_initial_value(capacity_predecessor(capacity_4, capacity_5), True)
        problem.set_initial_value(capacity_predecessor(capacity_5, capacity_6), True)

        problem.set_initial_value(road(city_loc_0, city_loc_2), True)
        problem.set_initial_value(road(city_loc_0, city_loc_4), True)
        problem.set_initial_value(road(city_loc_1, city_loc_4), True)
        problem.set_initial_value(road(city_loc_2, city_loc_0), True)
        problem.set_initial_value(road(city_loc_3, city_loc_4), True)
        problem.set_initial_value(road(city_loc_4, city_loc_0), True)
        problem.set_initial_value(road(city_loc_4, city_loc_1), True)
        problem.set_initial_value(road(city_loc_4, city_loc_3), True)

        problem.set_initial_value(at(package_0, city_loc_0), True)
        problem.set_initial_value(at(package_1, city_loc_3), True)
        problem.set_initial_value(at(package_2, city_loc_2), True)
        problem.set_initial_value(at(package_3, city_loc_1), True)
        problem.set_initial_value(at(package_4, city_loc_4), True)
        problem.set_initial_value(at(package_5, city_loc_1), True)

        problem.set_initial_value(at(truck_0, city_loc_3), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_6), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_2))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_2))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_4))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_3))  # Corresponds to PDDL task3
        t4 = gn.add(at(package_4, city_loc_0))  # Corresponds to PDDL task4
        t5 = gn.add(at(package_5, city_loc_3))  # Corresponds to PDDL task5

        # Add ordering constraints
        gn.add_ordering(t1, t2)  # (< task1 task2)
        gn.add_ordering(t4, t2)  # (< task4 task2)
        gn.add_ordering(t0, t2)  # (< task0 task2)
        gn.add_ordering(t2, t3)  # (< task2 task3)
        gn.add_ordering(t2, t5)  # (< task2 task5)

        problem.set_goal_network(gn)

    if problem_instance == 8:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)
        package_4 = problem.add_object("package_4", Package)
        package_5 = problem.add_object("package_5", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)
        capacity_4 = problem.add_object("capacity_4", CapacityNumber)
        capacity_5 = problem.add_object("capacity_5", CapacityNumber)
        capacity_6 = problem.add_object("capacity_6", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)
        city_loc_4 = problem.add_object("city_loc_4", Location)
        city_loc_5 = problem.add_object("city_loc_5", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)
        problem.set_initial_value(capacity_predecessor(capacity_3, capacity_4), True)
        problem.set_initial_value(capacity_predecessor(capacity_4, capacity_5), True)
        problem.set_initial_value(capacity_predecessor(capacity_5, capacity_6), True)

        problem.set_initial_value(road(city_loc_0, city_loc_2), True)
        problem.set_initial_value(road(city_loc_0, city_loc_4), True)
        problem.set_initial_value(road(city_loc_0, city_loc_5), True)
        problem.set_initial_value(road(city_loc_1, city_loc_4), True)
        problem.set_initial_value(road(city_loc_2, city_loc_0), True)
        problem.set_initial_value(road(city_loc_2, city_loc_5), True)
        problem.set_initial_value(road(city_loc_3, city_loc_5), True)
        problem.set_initial_value(road(city_loc_4, city_loc_0), True)
        problem.set_initial_value(road(city_loc_4, city_loc_1), True)
        problem.set_initial_value(road(city_loc_4, city_loc_4), True)
        problem.set_initial_value(road(city_loc_5, city_loc_0), True)
        problem.set_initial_value(road(city_loc_5, city_loc_2), True)
        problem.set_initial_value(road(city_loc_5, city_loc_3), True)

        problem.set_initial_value(at(package_0, city_loc_0), True)
        problem.set_initial_value(at(package_1, city_loc_4), True)
        problem.set_initial_value(at(package_2, city_loc_1), True)
        problem.set_initial_value(at(package_3, city_loc_5), True)
        problem.set_initial_value(at(package_4, city_loc_5), True)
        problem.set_initial_value(at(package_5, city_loc_0), True)

        problem.set_initial_value(at(truck_0, city_loc_0), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_6), True)

        # 3. Set Goal Network
        # The 'deliver' tasks in PDDL HTN typically correspond to 'at' goals for the package
        # at the specified location in the Unified Planning library's HTN representation.
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_1))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_5))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_3))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_4))  # Corresponds to PDDL task3
        t4 = gn.add(at(package_4, city_loc_4))  # Corresponds to PDDL task4
        t5 = gn.add(at(package_5, city_loc_5))  # Corresponds to PDDL task5

        # Add ordering constraints
        gn.add_ordering(t1, t0)  # (< task1 task0)
        gn.add_ordering(t5, t1)  # (< task5 task1)
        gn.add_ordering(t4, t2)  # (< task4 task2)
        gn.add_ordering(t3, t4)  # (< task3 task4)
        gn.add_ordering(t2, t5)  # (< task2 task5)

        problem.set_goal_network(gn)

    if problem_instance == 9:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)
        package_4 = problem.add_object("package_4", Package)
        package_5 = problem.add_object("package_5", Package)
        package_6 = problem.add_object("package_6", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)
        capacity_4 = problem.add_object("capacity_4", CapacityNumber)
        capacity_5 = problem.add_object("capacity_5", CapacityNumber)
        capacity_6 = problem.add_object("capacity_6", CapacityNumber)
        capacity_7 = problem.add_object("capacity_7", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)
        city_loc_4 = problem.add_object("city_loc_4", Location)
        city_loc_5 = problem.add_object("city_loc_5", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)
        problem.set_initial_value(capacity_predecessor(capacity_3, capacity_4), True)
        problem.set_initial_value(capacity_predecessor(capacity_4, capacity_5), True)
        problem.set_initial_value(capacity_predecessor(capacity_5, capacity_6), True)
        problem.set_initial_value(capacity_predecessor(capacity_6, capacity_7), True)

        problem.set_initial_value(road(city_loc_0, city_loc_5), True)
        problem.set_initial_value(road(city_loc_1, city_loc_1), True)
        problem.set_initial_value(road(city_loc_1, city_loc_4), True)
        problem.set_initial_value(road(city_loc_2, city_loc_4), True)
        problem.set_initial_value(road(city_loc_3, city_loc_5), True)
        problem.set_initial_value(road(city_loc_4, city_loc_1), True)
        problem.set_initial_value(road(city_loc_4, city_loc_2), True)
        problem.set_initial_value(road(city_loc_5, city_loc_0), True)
        problem.set_initial_value(road(city_loc_5, city_loc_3), True)

        problem.set_initial_value(at(package_0, city_loc_4), True)
        problem.set_initial_value(at(package_1, city_loc_4), True)
        problem.set_initial_value(at(package_2, city_loc_1), True)
        problem.set_initial_value(at(package_3, city_loc_2), True)
        problem.set_initial_value(at(package_4, city_loc_1), True)
        problem.set_initial_value(at(package_5, city_loc_4), True)
        problem.set_initial_value(at(package_6, city_loc_1), True)

        problem.set_initial_value(at(truck_0, city_loc_4), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_7), True)

        # 3. Set Goal Network
        # The 'deliver' tasks in PDDL HTN typically correspond to 'at' goals for the package
        # at the specified location in the Unified Planning library's HTN representation.
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_1))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_2))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_4))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_1))  # Corresponds to PDDL task3
        t4 = gn.add(at(package_4, city_loc_2))  # Corresponds to PDDL task4
        t5 = gn.add(at(package_5, city_loc_1))  # Corresponds to PDDL task5
        t6 = gn.add(at(package_6, city_loc_4))  # Corresponds to PDDL task6

        # Add ordering constraints
        gn.add_ordering(t5, t1)  # (< task5 task1)
        gn.add_ordering(t3, t2)  # (< task3 task2)
        gn.add_ordering(t4, t3)  # (< task4 task3)
        gn.add_ordering(t0, t4)  # (< task0 task4)
        gn.add_ordering(t2, t5)  # (< task2 task5)
        gn.add_ordering(t1, t6)  # (< task1 task6)

        problem.set_goal_network(gn)

    if problem_instance == 10:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)
        package_4 = problem.add_object("package_4", Package)
        package_5 = problem.add_object("package_5", Package)
        package_6 = problem.add_object("package_6", Package)
        package_7 = problem.add_object("package_7", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)
        capacity_4 = problem.add_object("capacity_4", CapacityNumber)
        capacity_5 = problem.add_object("capacity_5", CapacityNumber)
        capacity_6 = problem.add_object("capacity_6", CapacityNumber)
        capacity_7 = problem.add_object("capacity_7", CapacityNumber)
        capacity_8 = problem.add_object("capacity_8", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)
        city_loc_4 = problem.add_object("city_loc_4", Location)
        city_loc_5 = problem.add_object("city_loc_5", Location)
        city_loc_6 = problem.add_object("city_loc_6", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)
        problem.set_initial_value(capacity_predecessor(capacity_3, capacity_4), True)
        problem.set_initial_value(capacity_predecessor(capacity_4, capacity_5), True)
        problem.set_initial_value(capacity_predecessor(capacity_5, capacity_6), True)
        problem.set_initial_value(capacity_predecessor(capacity_6, capacity_7), True)
        problem.set_initial_value(capacity_predecessor(capacity_7, capacity_8), True)

        problem.set_initial_value(road(city_loc_0, city_loc_6), True)
        problem.set_initial_value(road(city_loc_1, city_loc_1), True)
        problem.set_initial_value(road(city_loc_1, city_loc_2), True)
        problem.set_initial_value(road(city_loc_2, city_loc_1), True)
        problem.set_initial_value(road(city_loc_2, city_loc_4), True)
        problem.set_initial_value(road(city_loc_2, city_loc_5), True)
        problem.set_initial_value(road(city_loc_3, city_loc_6), True)
        problem.set_initial_value(road(city_loc_4, city_loc_2), True)
        problem.set_initial_value(road(city_loc_5, city_loc_2), True)
        problem.set_initial_value(road(city_loc_6, city_loc_0), True)
        problem.set_initial_value(road(city_loc_6, city_loc_3), True)
        problem.set_initial_value(road(city_loc_6, city_loc_6), True)

        problem.set_initial_value(at(package_0, city_loc_6), True)
        problem.set_initial_value(at(package_1, city_loc_3), True)
        problem.set_initial_value(at(package_2, city_loc_0), True)
        problem.set_initial_value(at(package_3, city_loc_0), True)
        problem.set_initial_value(at(package_4, city_loc_3), True)
        problem.set_initial_value(at(package_5, city_loc_6), True)
        problem.set_initial_value(at(package_6, city_loc_6), True)
        problem.set_initial_value(at(package_7, city_loc_0), True)

        problem.set_initial_value(at(truck_0, city_loc_6), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_8), True)

        # 3. Set Goal Network
        # The 'deliver' tasks in PDDL HTN typically correspond to 'at' goals for the package
        # at the specified location in the Unified Planning library's HTN representation.
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_3))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_6))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_6))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_3))  # Corresponds to PDDL task3
        t4 = gn.add(at(package_4, city_loc_6))  # Corresponds to PDDL task4
        t5 = gn.add(at(package_5, city_loc_0))  # Corresponds to PDDL task5
        t6 = gn.add(at(package_6, city_loc_3))  # Corresponds to PDDL task6
        t7 = gn.add(at(package_7, city_loc_6))  # Corresponds to PDDL task7

        # Add ordering constraints
        gn.add_ordering(t3, t0)  # (< task3 task0)
        gn.add_ordering(t5, t1)  # (< task5 task1)
        gn.add_ordering(t6, t2)  # (< task6 task2)
        gn.add_ordering(t1, t4)  # (< task1 task4)
        gn.add_ordering(t0, t5)  # (< task0 task5)
        gn.add_ordering(t4, t6)  # (< task4 task6)
        gn.add_ordering(t2, t7)  # (< task2 task7)

        problem.set_goal_network(gn)

    if problem_instance == 11:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)
        truck_1 = problem.add_object("truck_1", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)

        problem.set_initial_value(road(city_loc_0, city_loc_1), True)
        problem.set_initial_value(road(city_loc_0, city_loc_3), True)
        problem.set_initial_value(road(city_loc_1, city_loc_0), True)
        problem.set_initial_value(road(city_loc_1, city_loc_2), True)
        problem.set_initial_value(road(city_loc_2, city_loc_1), True)
        problem.set_initial_value(road(city_loc_3, city_loc_0), True)

        problem.set_initial_value(at(package_0, city_loc_2), True)
        problem.set_initial_value(at(package_1, city_loc_2), True)
        problem.set_initial_value(at(package_2, city_loc_0), True)
        problem.set_initial_value(at(package_3, city_loc_0), True)

        problem.set_initial_value(at(truck_0, city_loc_0), True)
        problem.set_initial_value(at(truck_1, city_loc_1), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_2), True)
        problem.set_initial_value(current_capacity_level(truck_1, capacity_2), True)

        # 3. Set Goal Network
        # The 'deliver' tasks in PDDL HTN typically correspond to 'at' goals for the package
        # at the specified location in the Unified Planning library's HTN representation.
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_1))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_3))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_3))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_2))  # Corresponds to PDDL task3

        # Add ordering constraints
        gn.add_ordering(t2, t3)  # (< task2 task3)
        gn.add_ordering(t0, t1)  # (< task0 task1)

        problem.set_goal_network(gn)

    if problem_instance == 12:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)
        city_loc_4 = problem.add_object("city_loc_4", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)
        truck_1 = problem.add_object("truck_1", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)

        problem.set_initial_value(road(city_loc_0, city_loc_0), True)
        problem.set_initial_value(road(city_loc_0, city_loc_4), True)
        problem.set_initial_value(road(city_loc_1, city_loc_1), True)
        problem.set_initial_value(road(city_loc_1, city_loc_2), True)
        problem.set_initial_value(road(city_loc_1, city_loc_3), True)
        problem.set_initial_value(road(city_loc_1, city_loc_4), True)
        problem.set_initial_value(road(city_loc_2, city_loc_1), True)
        problem.set_initial_value(road(city_loc_2, city_loc_3), True)
        problem.set_initial_value(road(city_loc_3, city_loc_1), True)
        problem.set_initial_value(road(city_loc_3, city_loc_2), True)
        problem.set_initial_value(road(city_loc_4, city_loc_0), True)
        problem.set_initial_value(road(city_loc_4, city_loc_1), True)

        problem.set_initial_value(at(package_0, city_loc_3), True)
        problem.set_initial_value(at(package_1, city_loc_1), True)
        problem.set_initial_value(at(package_2, city_loc_3), True)
        problem.set_initial_value(at(package_3, city_loc_1), True)

        problem.set_initial_value(at(truck_0, city_loc_1), True)
        problem.set_initial_value(at(truck_1, city_loc_1), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_3), True)
        problem.set_initial_value(current_capacity_level(truck_1, capacity_1), True)

        # 3. Set Goal Network
        # The 'deliver' tasks in PDDL HTN typically correspond to 'at' goals for the package
        # at the specified location in the Unified Planning library's HTN representation.
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_0))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_2))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_2))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_3))  # Corresponds to PDDL task3

        # Add ordering constraints
        gn.add_ordering(t1, t2)  # (< task1 task2)
        gn.add_ordering(t3, t2)  # (< task3 task2)

        problem.set_goal_network(gn)

    if problem_instance == 13:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)
        package_4 = problem.add_object("package_4", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)
        capacity_4 = problem.add_object("capacity_4", CapacityNumber)
        capacity_5 = problem.add_object("capacity_5", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)
        city_loc_4 = problem.add_object("city_loc_4", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)
        truck_1 = problem.add_object("truck_1", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)

        problem.set_initial_value(road(city_loc_0, city_loc_1), True)
        problem.set_initial_value(road(city_loc_0, city_loc_2), True)
        problem.set_initial_value(road(city_loc_0, city_loc_3), True)
        problem.set_initial_value(road(city_loc_1, city_loc_0), True)
        problem.set_initial_value(road(city_loc_1, city_loc_4), True)
        problem.set_initial_value(road(city_loc_2, city_loc_0), True)
        problem.set_initial_value(road(city_loc_3, city_loc_0), True)
        problem.set_initial_value(road(city_loc_3, city_loc_3), True)
        problem.set_initial_value(road(city_loc_4, city_loc_1), True)

        problem.set_initial_value(at(package_0, city_loc_4), True)
        problem.set_initial_value(at(package_1, city_loc_2), True)
        problem.set_initial_value(at(package_2, city_loc_2), True)
        problem.set_initial_value(at(package_3, city_loc_1), True)
        problem.set_initial_value(at(package_4, city_loc_0), True)

        problem.set_initial_value(at(truck_0, city_loc_0), True)
        problem.set_initial_value(at(truck_1, city_loc_0), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_3), True)
        problem.set_initial_value(current_capacity_level(truck_1, capacity_2), True)

        # 3. Set Goal Network
        # The 'deliver' tasks in PDDL HTN typically correspond to 'at' goals for the package
        # at the specified location in the Unified Planning library's HTN representation.
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_0))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_4))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_4))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_3))  # Corresponds to PDDL task3
        t4 = gn.add(at(package_4, city_loc_4))  # Corresponds to PDDL task4

        # Add ordering constraints
        gn.add_ordering(t3, t0)  # (< task3 task0)

        problem.set_goal_network(gn)

    if problem_instance == 14:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)
        package_4 = problem.add_object("package_4", Package)
        package_5 = problem.add_object("package_5", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)
        capacity_4 = problem.add_object("capacity_4", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)
        city_loc_4 = problem.add_object("city_loc_4", Location)
        city_loc_5 = problem.add_object("city_loc_5", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)
        truck_1 = problem.add_object("truck_1", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)
        problem.set_initial_value(capacity_predecessor(capacity_3, capacity_4), True)

        problem.set_initial_value(road(city_loc_0, city_loc_5), True)
        problem.set_initial_value(road(city_loc_1, city_loc_3), True)
        problem.set_initial_value(road(city_loc_2, city_loc_2), True)
        problem.set_initial_value(road(city_loc_2, city_loc_4), True)
        problem.set_initial_value(road(city_loc_2, city_loc_5), True)
        problem.set_initial_value(road(city_loc_3, city_loc_1), True)
        problem.set_initial_value(road(city_loc_3, city_loc_5), True)
        problem.set_initial_value(road(city_loc_4, city_loc_2), True)
        problem.set_initial_value(road(city_loc_5, city_loc_0), True)
        problem.set_initial_value(road(city_loc_5, city_loc_2), True)
        problem.set_initial_value(road(city_loc_5, city_loc_3), True)
        problem.set_initial_value(road(city_loc_5, city_loc_5), True)

        problem.set_initial_value(at(package_0, city_loc_4), True)
        problem.set_initial_value(at(package_1, city_loc_2), True)
        problem.set_initial_value(at(package_2, city_loc_4), True)
        problem.set_initial_value(at(package_3, city_loc_0), True)
        problem.set_initial_value(at(package_4, city_loc_2), True)
        problem.set_initial_value(at(package_5, city_loc_1), True)

        problem.set_initial_value(at(truck_0, city_loc_0), True)
        problem.set_initial_value(at(truck_1, city_loc_0), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_4), True)
        problem.set_initial_value(current_capacity_level(truck_1, capacity_2), True)

        # 3. Set Goal Network
        # The 'deliver' tasks in PDDL HTN typically correspond to 'at' goals for the package
        # at the specified location in the Unified Planning library's HTN representation.
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_0))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_4))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_5))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_2))  # Corresponds to PDDL task3
        t4 = gn.add(at(package_4, city_loc_1))  # Corresponds to PDDL task4
        t5 = gn.add(at(package_5, city_loc_3))  # Corresponds to PDDL task5

        # Add ordering constraints
        gn.add_ordering(t3, t1)  # (< task3 task1)
        gn.add_ordering(t1, t2)  # (< task1 task2)
        gn.add_ordering(t1, t0)  # (< task1 task0)
        gn.add_ordering(t4, t5)  # (< task4 task5)

        problem.set_goal_network(gn)

    if problem_instance == 15:
        # 1. Define Objects
        package_0 = problem.add_object("package_0", Package)
        package_1 = problem.add_object("package_1", Package)
        package_2 = problem.add_object("package_2", Package)
        package_3 = problem.add_object("package_3", Package)
        package_4 = problem.add_object("package_4", Package)
        package_5 = problem.add_object("package_5", Package)
        package_6 = problem.add_object("package_6", Package)

        capacity_0 = problem.add_object("capacity_0", CapacityNumber)
        capacity_1 = problem.add_object("capacity_1", CapacityNumber)
        capacity_2 = problem.add_object("capacity_2", CapacityNumber)
        capacity_3 = problem.add_object("capacity_3", CapacityNumber)
        capacity_4 = problem.add_object("capacity_4", CapacityNumber)
        capacity_5 = problem.add_object("capacity_5", CapacityNumber)

        city_loc_0 = problem.add_object("city_loc_0", Location)
        city_loc_1 = problem.add_object("city_loc_1", Location)
        city_loc_2 = problem.add_object("city_loc_2", Location)
        city_loc_3 = problem.add_object("city_loc_3", Location)
        city_loc_4 = problem.add_object("city_loc_4", Location)
        city_loc_5 = problem.add_object("city_loc_5", Location)
        city_loc_6 = problem.add_object("city_loc_6", Location)

        truck_0 = problem.add_object("truck_0", Vehicle)
        truck_1 = problem.add_object("truck_1", Vehicle)

        # 2. Set Initial State
        problem.set_initial_value(capacity_predecessor(capacity_0, capacity_1), True)
        problem.set_initial_value(capacity_predecessor(capacity_1, capacity_2), True)
        problem.set_initial_value(capacity_predecessor(capacity_2, capacity_3), True)
        problem.set_initial_value(capacity_predecessor(capacity_3, capacity_4), True)
        problem.set_initial_value(capacity_predecessor(capacity_4, capacity_5), True)

        problem.set_initial_value(road(city_loc_0, city_loc_5), True)
        problem.set_initial_value(road(city_loc_1, city_loc_3), True)
        problem.set_initial_value(road(city_loc_2, city_loc_3), True)
        problem.set_initial_value(road(city_loc_2, city_loc_4), True)
        problem.set_initial_value(road(city_loc_2, city_loc_6), True)
        problem.set_initial_value(road(city_loc_3, city_loc_1), True)
        problem.set_initial_value(road(city_loc_3, city_loc_2), True)
        problem.set_initial_value(road(city_loc_3, city_loc_4), True)
        problem.set_initial_value(road(city_loc_4, city_loc_2), True)
        problem.set_initial_value(road(city_loc_4, city_loc_3), True)
        problem.set_initial_value(road(city_loc_4, city_loc_5), True)
        problem.set_initial_value(road(city_loc_5, city_loc_0), True)
        problem.set_initial_value(road(city_loc_5, city_loc_4), True)
        problem.set_initial_value(road(city_loc_6, city_loc_2), True)

        problem.set_initial_value(at(package_0, city_loc_6), True)
        problem.set_initial_value(at(package_1, city_loc_1), True)
        problem.set_initial_value(at(package_2, city_loc_6), True)
        problem.set_initial_value(at(package_3, city_loc_1), True)
        problem.set_initial_value(at(package_4, city_loc_4), True)
        problem.set_initial_value(at(package_5, city_loc_3), True)
        problem.set_initial_value(at(package_6, city_loc_4), True)

        problem.set_initial_value(at(truck_0, city_loc_6), True)
        problem.set_initial_value(at(truck_1, city_loc_2), True)
        problem.set_initial_value(current_capacity_level(truck_0, capacity_5), True)
        problem.set_initial_value(current_capacity_level(truck_1, capacity_5), True)

        # 3. Set Goal Network
        # The 'deliver' tasks in PDDL HTN typically correspond to 'at' goals for the package
        # at the specified location in the Unified Planning library's HTN representation.
        gn = PartialOrderGoalNetwork()
        t0 = gn.add(at(package_0, city_loc_2))  # Corresponds to PDDL task0
        t1 = gn.add(at(package_1, city_loc_6))  # Corresponds to PDDL task1
        t2 = gn.add(at(package_2, city_loc_5))  # Corresponds to PDDL task2
        t3 = gn.add(at(package_3, city_loc_2))  # Corresponds to PDDL task3
        t4 = gn.add(at(package_4, city_loc_6))  # Corresponds to PDDL task4
        t5 = gn.add(at(package_5, city_loc_0))  # Corresponds to PDDL task5
        t6 = gn.add(at(package_6, city_loc_2))  # Corresponds to PDDL task6

        # Add ordering constraints
        gn.add_ordering(t3, t0)  # (< task3 task0)
        gn.add_ordering(t2, t1)  # (< task2 task1)
        gn.add_ordering(t0, t2)  # (< task0 task2)
        gn.add_ordering(t6, t4)  # (< task6 task4)
        gn.add_ordering(t1, t5)  # (< task1 task5)
        gn.add_ordering(t5, t6)  # (< task5 task6)

        problem.set_goal_network(gn)

    return problem
