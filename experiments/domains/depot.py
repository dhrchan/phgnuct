from unified_planning.shortcuts import *
from unified_planning.model.phgn import *
from unified_planning.model.phgn.goal_network import PartialOrderGoalNetwork


def depot(problem_instance: int = 0):
    # Initialize the problem
    problem = PHGNProblem()

    # 1. Define UserTypes (with hierarchy)

    Object = UserType("Object")

    Place = UserType("Place", father=Object)
    Locatable = UserType("Locatable", father=Object)

    Depot = UserType("Depot", father=Place)
    Distributor = UserType("Distributor", father=Place)

    Truck = UserType("Truck", father=Locatable)
    Hoist = UserType("Hoist", father=Locatable)
    Surface = UserType("Surface", father=Locatable)

    Pallet = UserType("Pallet", father=Surface)
    Crate = UserType("Crate", father=Surface)

    # 2. Define Fluents (Predicates)

    # (at ?x - locatable ?y - place)
    at = problem.add_fluent(
        "at", BoolType(), default_initial_value=False, x=Locatable, y=Place
    )

    # (on ?x - crate ?y - surface)
    on = problem.add_fluent(
        "on", BoolType(), default_initial_value=False, x=Crate, y=Surface
    )

    # (in ?x - crate ?y - truck)
    in_truck = problem.add_fluent(
        "in_truck", BoolType(), default_initial_value=False, x=Crate, y=Truck
    )

    # (lifting ?x - hoist ?y - crate)
    lifting = problem.add_fluent(
        "lifting", BoolType(), default_initial_value=False, x=Hoist, y=Crate
    )

    # (available ?x - hoist)
    available = problem.add_fluent(
        "available", BoolType(), default_initial_value=False, x=Hoist
    )

    # (clear ?x - surface)
    clear = problem.add_fluent(
        "clear", BoolType(), default_initial_value=False, x=Surface
    )

    # (supervisor_needed ?x - hoist ?y - crate ?z - truck ?p - place)
    supervisor_needed = problem.add_fluent(
        "supervisor_needed",
        BoolType(),
        default_initial_value=False,
        x=Hoist,
        y=Crate,
        z=Truck,
        p=Place,
    )

    # 3. Define Actions

    # (:action Drive
    #   :parameters (?x - truck ?y - place ?z - place)
    #   :precondition (and (at ?x ?y))
    #   :effect (and (not (at ?x ?y)) (at ?x ?z))
    # )
    drive = InstantaneousAction("drive", x=Truck, y=Place, z=Place)
    drive.add_precondition(at(drive.x, drive.y))
    drive.add_effect(at(drive.x, drive.y), False)
    drive.add_effect(at(drive.x, drive.z), True)
    problem.add_action(drive)

    # (:action Lift
    #   :parameters (?x - hoist ?y - crate ?z - surface ?p - place)
    #   :precondition (and
    #     (at ?x ?p) (available ?x) (at ?y ?p) (on ?y ?z) (clear ?y)
    #   )
    #   :effect (and
    #     (not (at ?y ?p)) (lifting ?x ?y) (not (clear ?y))
    #     (not (available ?x)) (clear ?z) (not (on ?y ?z))
    #   )
    # )
    lift = InstantaneousAction("lift", x=Hoist, y=Crate, z=Surface, p=Place)
    lift.add_precondition(at(lift.x, lift.p))
    lift.add_precondition(available(lift.x))
    lift.add_precondition(at(lift.y, lift.p))
    lift.add_precondition(on(lift.y, lift.z))
    lift.add_precondition(clear(lift.y))
    lift.add_effect(at(lift.y, lift.p), False)
    lift.add_effect(lifting(lift.x, lift.y), True)
    lift.add_effect(clear(lift.y), False)
    lift.add_effect(available(lift.x), False)
    lift.add_effect(clear(lift.z), True)
    lift.add_effect(on(lift.y, lift.z), False)
    problem.add_action(lift)

    # (:action Drop
    #   :parameters (?x - hoist ?y - crate ?z - surface ?p - place)
    #   :precondition (and
    #     (at ?x ?p) (at ?z ?p) (clear ?z) (lifting ?x ?y)
    #   )
    #   :effect (and
    #     (available ?x) (not (lifting ?x ?y)) (at ?y ?p)
    #     (not (clear ?z)) (clear ?y) (on ?y ?z)
    #   )
    # )
    drop = InstantaneousAction("drop", x=Hoist, y=Crate, z=Surface, p=Place)
    drop.add_precondition(at(drop.x, drop.p))
    drop.add_precondition(at(drop.z, drop.p))
    drop.add_precondition(clear(drop.z))
    drop.add_precondition(lifting(drop.x, drop.y))
    drop.add_effect(available(drop.x), True)
    drop.add_effect(lifting(drop.x, drop.y), False)
    drop.add_effect(at(drop.y, drop.p), True)
    drop.add_effect(clear(drop.z), False)
    drop.add_effect(clear(drop.y), True)
    drop.add_effect(on(drop.y, drop.z), True)
    problem.add_action(drop)

    # (:action Unsupervised_Load
    #   :parameters (?x - hoist ?y - crate ?z - truck ?p - place)
    #   :precondition (and
    #     (at ?x ?p) (at ?z ?p) (lifting ?x ?y)
    #   )
    #   :effect ( probabilistic
    #     0.5 (and (not (lifting ?x ?y)) (in ?y ?z) (available ?x))
    #     0.5 (and (supervisor_needed ?x ?y ?z ?p))
    #   )
    # )
    unsupervised_load = ProbabilisticAction(
        "unsupervised_load", x=Hoist, y=Crate, z=Truck, p=Place
    )
    unsupervised_load.add_precondition(at(unsupervised_load.x, unsupervised_load.p))
    unsupervised_load.add_precondition(at(unsupervised_load.z, unsupervised_load.p))
    unsupervised_load.add_precondition(
        lifting(unsupervised_load.x, unsupervised_load.y)
    )

    # Outcome 1
    unsupervised_load.add_outcome("success", 0.5)
    unsupervised_load.add_effect(
        "success", lifting(unsupervised_load.x, unsupervised_load.y), False
    )
    unsupervised_load.add_effect(
        "success", in_truck(unsupervised_load.y, unsupervised_load.z), True
    )
    unsupervised_load.add_effect("success", available(unsupervised_load.x), True)

    # Outcome 2
    unsupervised_load.add_outcome("failure", 0.5)
    unsupervised_load.add_effect(
        "failure",
        supervisor_needed(
            unsupervised_load.x,
            unsupervised_load.y,
            unsupervised_load.z,
            unsupervised_load.p,
        ),
        True,
    )
    problem.add_action(unsupervised_load)

    # (:action Supervised_Load
    #   :parameters (?x - hoist ?y - crate ?z - truck ?p - place)
    #   :precondition (and
    #     (at ?x ?p) (at ?z ?p) (lifting ?x ?y) (supervisor_needed ?x ?y ?z ?p)
    #   )
    #   :effect (and (not (lifting ?x ?y)) (in ?y ?z) (available ?x) (not (supervisor_needed ?x ?y ?z ?p)))
    # )
    supervised_load = InstantaneousAction(
        "supervised_load", x=Hoist, y=Crate, z=Truck, p=Place
    )
    supervised_load.add_precondition(at(supervised_load.x, supervised_load.p))
    supervised_load.add_precondition(at(supervised_load.z, supervised_load.p))
    supervised_load.add_precondition(lifting(supervised_load.x, supervised_load.y))
    supervised_load.add_precondition(
        supervisor_needed(
            supervised_load.x, supervised_load.y, supervised_load.z, supervised_load.p
        )
    )
    supervised_load.add_effect(lifting(supervised_load.x, supervised_load.y), False)
    supervised_load.add_effect(in_truck(supervised_load.y, supervised_load.z), True)
    supervised_load.add_effect(available(supervised_load.x), True)
    supervised_load.add_effect(
        supervisor_needed(
            supervised_load.x, supervised_load.y, supervised_load.z, supervised_load.p
        ),
        False,
    )
    problem.add_action(supervised_load)

    # (:action Unsupervised_Unload
    #   :parameters (?x - hoist ?y - crate ?z - truck ?p - place)
    #   :precondition (and
    #     (at ?x ?p) (at ?z ?p) (available ?x) (in ?y ?z)
    #   )
    #   :effect ( probabilistic
    #     0.5 (and (not (in ?y ?z)) (not (available ?x)) (lifting ?x ?y))
    #     0.5 (and (supervisor_needed ?x ?y ?z ?p))
    #   )
    # )
    unsupervised_unload = ProbabilisticAction(
        "unsupervised_unload", x=Hoist, y=Crate, z=Truck, p=Place
    )
    unsupervised_unload.add_precondition(
        at(unsupervised_unload.x, unsupervised_unload.p)
    )
    unsupervised_unload.add_precondition(
        at(unsupervised_unload.z, unsupervised_unload.p)
    )
    unsupervised_unload.add_precondition(available(unsupervised_unload.x))
    unsupervised_unload.add_precondition(
        in_truck(unsupervised_unload.y, unsupervised_unload.z)
    )

    # Outcome 1
    unsupervised_unload.add_outcome("success", 0.5)
    unsupervised_unload.add_effect(
        "success", in_truck(unsupervised_unload.y, unsupervised_unload.z), False
    )
    unsupervised_unload.add_effect("success", available(unsupervised_unload.x), False)
    unsupervised_unload.add_effect(
        "success", lifting(unsupervised_unload.x, unsupervised_unload.y), True
    )

    # Outcome 2
    unsupervised_unload.add_outcome("failure", 0.5)
    unsupervised_unload.add_effect(
        "failure",
        supervisor_needed(
            unsupervised_unload.x,
            unsupervised_unload.y,
            unsupervised_unload.z,
            unsupervised_unload.p,
        ),
        True,
    )
    problem.add_action(unsupervised_unload)

    # (:action Supervised_Unload
    #   :parameters (?x - hoist ?y - crate ?z - truck ?p - place)
    #   :precondition (and
    #     (at ?x ?p) (at ?z ?p) (available ?x) (in ?y ?z) (supervisor_needed ?x ?y ?z ?p)
    #   )
    #   :effect (and (not (in ?y ?z)) (not (available ?x)) (lifting ?x ?y) (not (supervisor_needed ?x ?y ?z ?p)))
    # )
    supervised_unload = InstantaneousAction(
        "supervised_unload", x=Hoist, y=Crate, z=Truck, p=Place
    )
    supervised_unload.add_precondition(at(supervised_unload.x, supervised_unload.p))
    supervised_unload.add_precondition(at(supervised_unload.z, supervised_unload.p))
    supervised_unload.add_precondition(available(supervised_unload.x))
    supervised_unload.add_precondition(
        in_truck(supervised_unload.y, supervised_unload.z)
    )
    supervised_unload.add_precondition(
        supervisor_needed(
            supervised_unload.x,
            supervised_unload.y,
            supervised_unload.z,
            supervised_unload.p,
        )
    )
    supervised_unload.add_effect(
        in_truck(supervised_unload.y, supervised_unload.z), False
    )
    supervised_unload.add_effect(available(supervised_unload.x), False)
    supervised_unload.add_effect(
        lifting(supervised_unload.x, supervised_unload.y), True
    )
    supervised_unload.add_effect(
        supervisor_needed(
            supervised_unload.x,
            supervised_unload.y,
            supervised_unload.z,
            supervised_unload.p,
        ),
        False,
    )
    problem.add_action(supervised_unload)

    put_on_1 = PHGNMethod(
        "put_on_1",
        c=Crate,
        s2=Surface,
        p=Place,
        h=Hoist,
    )
    put_on_1.add_precondition(at(put_on_1.c, put_on_1.p))
    put_on_1.add_precondition(at(put_on_1.s2, put_on_1.p))
    put_on_1.add_precondition(at(put_on_1.h, put_on_1.p))
    gn = PartialOrderGoalNetwork()
    gn.add(
        And(clear(put_on_1.c), clear(put_on_1.s2)),
        lifting(put_on_1.h, put_on_1.c),
        on(put_on_1.c, put_on_1.s2),
    )
    put_on_1.set_goal_network(gn)
    problem.add_method(put_on_1)

    put_on_2 = PHGNMethod(
        "put_on_2",
        c=Crate,
        s2=Surface,
        p=Place,
        t=Truck,
        h=Hoist,
    )
    put_on_2.add_precondition(in_truck(put_on_2.c, put_on_2.t))
    gn = PartialOrderGoalNetwork()
    gn.add(
        at(put_on_2.t, put_on_2.p),
        clear(put_on_2.s2),
        lifting(put_on_2.h, put_on_2.c),
        on(put_on_2.c, put_on_2.s2),
    )
    put_on_2.set_goal_network(gn)
    problem.add_method(put_on_2)

    put_on_3 = PHGNMethod(
        "put_on_3",
        c=Crate,
        s2=Surface,
        p1=Place,
        p2=Place,
        t=Truck,
    )
    put_on_3.add_precondition(at(put_on_3.c, put_on_3.p1))
    put_on_3.add_precondition(at(put_on_3.s2, put_on_3.p2))
    put_on_3.add_precondition(Not(Equals(put_on_3.p1, put_on_3.p2)))
    gn = PartialOrderGoalNetwork()
    gn.add(
        in_truck(put_on_3.c, put_on_3.t),
        And(in_truck(put_on_3.c, put_on_3.t), at(put_on_3.t, put_on_3.p2)),
        on(put_on_3.c, put_on_3.s2),
    )
    put_on_3.set_goal_network(gn)
    problem.add_method(put_on_3)

    do_clear = PHGNMethod(
        "do_clear",
        s1=Surface,
        c=Crate,
    )
    do_clear.add_precondition(Not(clear(do_clear.s1)))
    do_clear.add_precondition(on(do_clear.c, do_clear.s1))
    gn = PartialOrderGoalNetwork()
    gn.add(
        clear(do_clear.c),
        clear(do_clear.s1),
    )
    do_clear.set_goal_network(gn)
    problem.add_method(do_clear)

    lift_crate = PHGNMethod(
        "lift_crate",
        c=Crate,
        p=Place,
        h=Hoist,
        t=Truck,
    )
    lift_crate.add_precondition(in_truck(lift_crate.c, lift_crate.t))
    lift_crate.add_precondition(at(lift_crate.h, lift_crate.p))
    gn = PartialOrderGoalNetwork()
    gn.add(
        at(lift_crate.t, lift_crate.p),
        lifting(lift_crate.h, lift_crate.c),
    )
    lift_crate.set_goal_network(gn)
    problem.add_method(lift_crate)

    load_truck = PHGNMethod(
        "load_truck",
        c=Crate,
        s=Surface,
        p=Place,
        t=Truck,
        h=Hoist,
    )
    load_truck.add_precondition(at(load_truck.c, load_truck.p))
    load_truck.add_precondition(at(load_truck.s, load_truck.p))
    load_truck.add_precondition(on(load_truck.c, load_truck.s))
    load_truck.add_precondition(at(load_truck.h, load_truck.p))
    gn = PartialOrderGoalNetwork()
    gn.add(
        at(load_truck.t, load_truck.p),
        clear(load_truck.c),
        lifting(load_truck.h, load_truck.c),
        in_truck(load_truck.c, load_truck.t),
    )
    load_truck.set_goal_network(gn)
    problem.add_method(load_truck)

    unload_truck = PHGNMethod(
        "unload_truck",
        c=Crate,
        s=Surface,
        p=Place,
        t=Truck,
        h=Hoist,
    )
    unload_truck.add_precondition(in_truck(unload_truck.c, unload_truck.t))
    unload_truck.add_precondition(at(unload_truck.t, unload_truck.p))
    unload_truck.add_precondition(at(unload_truck.h, unload_truck.p))
    unload_truck.add_precondition(at(unload_truck.s, unload_truck.p))
    gn = PartialOrderGoalNetwork()
    gn.add(
        clear(unload_truck.s),
        lifting(unload_truck.h, unload_truck.c),
        on(unload_truck.c, unload_truck.s),
    )
    unload_truck.set_goal_network(gn)
    problem.add_method(unload_truck)

    if problem_instance == 1:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate1), True)
        problem.set_initial_value(at(pallet1, distributor0), True)
        problem.set_initial_value(clear(crate0), True)
        problem.set_initial_value(at(pallet2, distributor1), True)
        problem.set_initial_value(clear(pallet2), True)
        problem.set_initial_value(at(truck0, distributor1), True)
        problem.set_initial_value(at(truck1, depot0), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, distributor0), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, distributor1), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(crate0, distributor0), True)
        problem.set_initial_value(on(crate0, pallet1), True)
        problem.set_initial_value(at(crate1, depot0), True)
        problem.set_initial_value(on(crate1, pallet0), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(on(crate1, pallet1), on(crate0, pallet2))
        problem.set_goal_network(gn)

    if problem_instance == 2:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate0), True)
        problem.set_initial_value(at(pallet1, distributor0), True)
        problem.set_initial_value(clear(crate3), True)
        problem.set_initial_value(at(pallet2, distributor1), True)
        problem.set_initial_value(clear(crate2), True)
        problem.set_initial_value(at(truck0, depot0), True)
        problem.set_initial_value(at(truck1, depot0), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, distributor0), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, distributor1), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(crate0, depot0), True)
        problem.set_initial_value(on(crate0, pallet0), True)
        problem.set_initial_value(at(crate1, distributor1), True)
        problem.set_initial_value(on(crate1, pallet2), True)
        problem.set_initial_value(at(crate2, distributor1), True)
        problem.set_initial_value(on(crate2, crate1), True)
        problem.set_initial_value(at(crate3, distributor0), True)
        problem.set_initial_value(on(crate3, pallet1), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate3, pallet1),
            on(crate2, pallet0),
            on(crate1, crate3),
            on(crate0, pallet2),
        )
        problem.set_goal_network(gn)

    if problem_instance == 3:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate1), True)
        problem.set_initial_value(at(pallet1, distributor0), True)
        problem.set_initial_value(clear(crate4), True)
        problem.set_initial_value(at(pallet2, distributor1), True)
        problem.set_initial_value(clear(crate5), True)
        problem.set_initial_value(at(truck0, depot0), True)
        problem.set_initial_value(at(truck1, distributor0), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, distributor0), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, distributor1), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(crate0, distributor0), True)
        problem.set_initial_value(on(crate0, pallet1), True)
        problem.set_initial_value(at(crate1, depot0), True)
        problem.set_initial_value(on(crate1, pallet0), True)
        problem.set_initial_value(at(crate2, distributor1), True)
        problem.set_initial_value(on(crate2, pallet2), True)
        problem.set_initial_value(at(crate3, distributor0), True)
        problem.set_initial_value(on(crate3, crate0), True)
        problem.set_initial_value(at(crate4, distributor0), True)
        problem.set_initial_value(on(crate4, crate3), True)
        problem.set_initial_value(at(crate5, distributor1), True)
        problem.set_initial_value(on(crate5, crate2), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate1, pallet2),
            on(crate2, pallet0),
            on(crate4, pallet1),
            on(crate0, crate1),
            on(crate3, crate2),
            on(crate5, crate0),
        )
        problem.set_goal_network(gn)

    if problem_instance == 4:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        crate6 = problem.add_object("crate6", Crate)
        crate7 = problem.add_object("crate7", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate7), True)
        problem.set_initial_value(at(pallet1, distributor0), True)
        problem.set_initial_value(clear(crate2), True)
        problem.set_initial_value(at(pallet2, distributor1), True)
        problem.set_initial_value(clear(crate6), True)
        problem.set_initial_value(at(truck0, distributor1), True)
        problem.set_initial_value(at(truck1, distributor1), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, distributor0), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, distributor1), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(crate0, depot0), True)
        problem.set_initial_value(on(crate0, pallet0), True)
        problem.set_initial_value(at(crate1, depot0), True)
        problem.set_initial_value(on(crate1, crate0), True)
        problem.set_initial_value(at(crate2, distributor0), True)
        problem.set_initial_value(on(crate2, pallet1), True)
        problem.set_initial_value(at(crate3, distributor1), True)
        problem.set_initial_value(on(crate3, pallet2), True)
        problem.set_initial_value(at(crate4, depot0), True)
        problem.set_initial_value(on(crate4, crate1), True)
        problem.set_initial_value(at(crate5, distributor1), True)
        problem.set_initial_value(on(crate5, crate3), True)
        problem.set_initial_value(at(crate6, distributor1), True)
        problem.set_initial_value(on(crate6, crate5), True)
        problem.set_initial_value(at(crate7, depot0), True)
        problem.set_initial_value(on(crate7, crate4), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate7, pallet0),
            on(crate6, pallet1),
            on(crate5, pallet2),
            on(crate4, crate7),
            on(crate2, crate6),
            on(crate0, crate4),
        )
        problem.set_goal_network(gn)

    if problem_instance == 5:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        crate6 = problem.add_object("crate6", Crate)
        crate7 = problem.add_object("crate7", Crate)
        crate8 = problem.add_object("crate8", Crate)
        crate9 = problem.add_object("crate9", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate4), True)
        problem.set_initial_value(at(pallet1, distributor0), True)
        problem.set_initial_value(clear(crate8), True)
        problem.set_initial_value(at(pallet2, distributor1), True)
        problem.set_initial_value(clear(crate9), True)
        problem.set_initial_value(at(truck0, depot0), True)
        problem.set_initial_value(at(truck1, distributor0), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, distributor0), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, distributor1), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(crate0, distributor1), True)
        problem.set_initial_value(on(crate0, pallet2), True)
        problem.set_initial_value(at(crate1, depot0), True)
        problem.set_initial_value(on(crate1, pallet0), True)
        problem.set_initial_value(at(crate2, distributor1), True)
        problem.set_initial_value(on(crate2, crate0), True)
        problem.set_initial_value(at(crate3, depot0), True)
        problem.set_initial_value(on(crate3, crate1), True)
        problem.set_initial_value(at(crate4, depot0), True)
        problem.set_initial_value(on(crate4, crate3), True)
        problem.set_initial_value(at(crate5, distributor1), True)
        problem.set_initial_value(on(crate5, crate2), True)
        problem.set_initial_value(at(crate6, distributor0), True)
        problem.set_initial_value(on(crate6, pallet1), True)
        problem.set_initial_value(at(crate7, distributor0), True)
        problem.set_initial_value(on(crate7, crate6), True)
        problem.set_initial_value(at(crate8, distributor0), True)
        problem.set_initial_value(on(crate8, crate7), True)
        problem.set_initial_value(at(crate9, distributor1), True)
        problem.set_initial_value(on(crate9, crate5), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate9, pallet0),
            on(crate3, pallet2),
            on(crate1, pallet1),
            on(crate6, crate9),
            on(crate4, crate6),
            on(crate5, crate4),
            on(crate0, crate5),
            on(crate2, crate0),
            on(crate7, crate1),
            on(crate8, crate3),
        )
        problem.set_goal_network(gn)

    if problem_instance == 6:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        crate6 = problem.add_object("crate6", Crate)
        crate7 = problem.add_object("crate7", Crate)
        crate8 = problem.add_object("crate8", Crate)
        crate9 = problem.add_object("crate9", Crate)
        crate10 = problem.add_object("crate10", Crate)
        crate11 = problem.add_object("crate11", Crate)
        crate12 = problem.add_object("crate12", Crate)
        crate13 = problem.add_object("crate13", Crate)
        crate14 = problem.add_object("crate14", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate11), True)
        problem.set_initial_value(at(pallet1, distributor0), True)
        problem.set_initial_value(clear(crate14), True)
        problem.set_initial_value(at(pallet2, distributor1), True)
        problem.set_initial_value(clear(crate10), True)
        problem.set_initial_value(at(truck0, distributor1), True)
        problem.set_initial_value(at(truck1, depot0), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, distributor0), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, distributor1), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(crate0, distributor1), True)
        problem.set_initial_value(on(crate0, pallet2), True)
        problem.set_initial_value(at(crate1, depot0), True)
        problem.set_initial_value(on(crate1, pallet0), True)
        problem.set_initial_value(at(crate2, distributor1), True)
        problem.set_initial_value(on(crate2, crate0), True)
        problem.set_initial_value(at(crate3, distributor0), True)
        problem.set_initial_value(on(crate3, pallet1), True)
        problem.set_initial_value(at(crate4, distributor0), True)
        problem.set_initial_value(on(crate4, crate3), True)
        problem.set_initial_value(at(crate5, distributor1), True)
        problem.set_initial_value(on(crate5, crate2), True)
        problem.set_initial_value(at(crate6, depot0), True)
        problem.set_initial_value(on(crate6, crate1), True)
        problem.set_initial_value(at(crate7, distributor0), True)
        problem.set_initial_value(on(crate7, crate4), True)
        problem.set_initial_value(at(crate8, distributor0), True)
        problem.set_initial_value(on(crate8, crate7), True)
        problem.set_initial_value(at(crate9, distributor0), True)
        problem.set_initial_value(on(crate9, crate8), True)
        problem.set_initial_value(at(crate10, distributor1), True)
        problem.set_initial_value(on(crate10, crate5), True)
        problem.set_initial_value(at(crate11, depot0), True)
        problem.set_initial_value(on(crate11, crate6), True)
        problem.set_initial_value(at(crate12, distributor0), True)
        problem.set_initial_value(on(crate12, crate9), True)
        problem.set_initial_value(at(crate13, distributor0), True)
        problem.set_initial_value(on(crate13, crate12), True)
        problem.set_initial_value(at(crate14, distributor0), True)
        problem.set_initial_value(on(crate14, crate13), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate8, pallet0),
            on(crate0, crate8),
            on(crate5, crate0),
            on(crate11, crate5),
            on(crate4, crate11),
            on(crate10, crate4),
            on(crate9, pallet1),
            on(crate1, crate9),
            on(crate2, crate1),
            on(crate12, pallet2),
            on(crate3, crate12),
        )
        problem.set_goal_network(gn)

    if problem_instance == 7:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        pallet3 = problem.add_object("pallet3", Pallet)
        pallet4 = problem.add_object("pallet4", Pallet)
        pallet5 = problem.add_object("pallet5", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate5), True)
        problem.set_initial_value(at(pallet1, distributor0), True)
        problem.set_initial_value(clear(pallet1), True)
        problem.set_initial_value(at(pallet2, distributor1), True)
        problem.set_initial_value(clear(crate3), True)
        problem.set_initial_value(at(pallet3, distributor0), True)
        problem.set_initial_value(clear(pallet3), True)
        problem.set_initial_value(at(pallet4, distributor0), True)
        problem.set_initial_value(clear(crate4), True)
        problem.set_initial_value(at(pallet5, distributor1), True)
        problem.set_initial_value(clear(crate1), True)
        problem.set_initial_value(at(truck0, distributor1), True)
        problem.set_initial_value(at(truck1, depot0), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, distributor0), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, distributor1), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(crate0, distributor0), True)
        problem.set_initial_value(on(crate0, pallet4), True)
        problem.set_initial_value(at(crate1, distributor1), True)
        problem.set_initial_value(on(crate1, pallet5), True)
        problem.set_initial_value(at(crate2, distributor1), True)
        problem.set_initial_value(on(crate2, pallet2), True)
        problem.set_initial_value(at(crate3, distributor1), True)
        problem.set_initial_value(on(crate3, crate2), True)
        problem.set_initial_value(at(crate4, distributor0), True)
        problem.set_initial_value(on(crate4, crate0), True)
        problem.set_initial_value(at(crate5, depot0), True)
        problem.set_initial_value(on(crate5, pallet0), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate3, pallet1),
            on(crate0, pallet3),
            on(crate4, pallet5),
            on(crate1, crate4),
            on(crate5, crate1),
        )
        problem.set_goal_network(gn)

    if problem_instance == 8:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        pallet3 = problem.add_object("pallet3", Pallet)
        pallet4 = problem.add_object("pallet4", Pallet)
        pallet5 = problem.add_object("pallet5", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        crate6 = problem.add_object("crate6", Crate)
        crate7 = problem.add_object("crate7", Crate)
        crate8 = problem.add_object("crate8", Crate)
        crate9 = problem.add_object("crate9", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate2), True)
        problem.set_initial_value(at(pallet1, distributor0), True)
        problem.set_initial_value(clear(crate6), True)
        problem.set_initial_value(at(pallet2, distributor1), True)
        problem.set_initial_value(clear(crate9), True)
        problem.set_initial_value(at(pallet3, distributor1), True)
        problem.set_initial_value(clear(crate7), True)
        problem.set_initial_value(at(pallet4, distributor0), True)
        problem.set_initial_value(clear(crate0), True)
        problem.set_initial_value(at(pallet5, distributor0), True)
        problem.set_initial_value(clear(crate8), True)
        problem.set_initial_value(at(truck0, distributor0), True)
        problem.set_initial_value(at(truck1, distributor0), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, distributor0), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, distributor1), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(crate0, distributor0), True)
        problem.set_initial_value(on(crate0, pallet4), True)
        problem.set_initial_value(at(crate1, distributor0), True)
        problem.set_initial_value(on(crate1, pallet1), True)
        problem.set_initial_value(at(crate2, depot0), True)
        problem.set_initial_value(on(crate2, pallet0), True)
        problem.set_initial_value(at(crate3, distributor0), True)
        problem.set_initial_value(on(crate3, pallet5), True)
        problem.set_initial_value(at(crate4, distributor1), True)
        problem.set_initial_value(on(crate4, pallet3), True)
        problem.set_initial_value(at(crate5, distributor0), True)
        problem.set_initial_value(on(crate5, crate1), True)
        problem.set_initial_value(at(crate6, distributor0), True)
        problem.set_initial_value(on(crate6, crate5), True)
        problem.set_initial_value(at(crate7, distributor1), True)
        problem.set_initial_value(on(crate7, crate4), True)
        problem.set_initial_value(at(crate8, distributor0), True)
        problem.set_initial_value(on(crate8, crate3), True)
        problem.set_initial_value(at(crate9, distributor1), True)
        problem.set_initial_value(on(crate9, pallet2), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate9, pallet0),
            on(crate7, pallet1),
            on(crate6, pallet2),
            on(crate0, pallet3),
            on(crate8, pallet4),
            on(crate3, crate8),
            on(crate1, crate0),
        )
        problem.set_goal_network(gn)

    if problem_instance == 9:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        pallet3 = problem.add_object("pallet3", Pallet)
        pallet4 = problem.add_object("pallet4", Pallet)
        pallet5 = problem.add_object("pallet5", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        crate6 = problem.add_object("crate6", Crate)
        crate7 = problem.add_object("crate7", Crate)
        crate8 = problem.add_object("crate8", Crate)
        crate9 = problem.add_object("crate9", Crate)
        crate10 = problem.add_object("crate10", Crate)
        crate11 = problem.add_object("crate11", Crate)
        crate12 = problem.add_object("crate12", Crate)
        crate13 = problem.add_object("crate13", Crate)
        crate14 = problem.add_object("crate14", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate2), True)
        problem.set_initial_value(at(pallet1, distributor0), True)
        problem.set_initial_value(clear(crate14), True)
        problem.set_initial_value(at(pallet2, distributor1), True)
        problem.set_initial_value(clear(crate13), True)
        problem.set_initial_value(at(pallet3, distributor1), True)
        problem.set_initial_value(clear(crate10), True)
        problem.set_initial_value(at(pallet4, distributor0), True)
        problem.set_initial_value(clear(crate12), True)
        problem.set_initial_value(at(pallet5, depot0), True)
        problem.set_initial_value(clear(crate8), True)
        problem.set_initial_value(at(truck0, distributor0), True)
        problem.set_initial_value(at(truck1, distributor0), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, distributor0), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, distributor1), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(crate0, distributor1), True)
        problem.set_initial_value(on(crate0, pallet2), True)
        problem.set_initial_value(at(crate1, depot0), True)
        problem.set_initial_value(on(crate1, pallet0), True)
        problem.set_initial_value(at(crate2, depot0), True)
        problem.set_initial_value(on(crate2, crate1), True)
        problem.set_initial_value(at(crate3, distributor0), True)
        problem.set_initial_value(on(crate3, pallet1), True)
        problem.set_initial_value(at(crate4, distributor1), True)
        problem.set_initial_value(on(crate4, crate0), True)
        problem.set_initial_value(at(crate5, distributor1), True)
        problem.set_initial_value(on(crate5, pallet3), True)
        problem.set_initial_value(at(crate6, distributor0), True)
        problem.set_initial_value(on(crate6, crate3), True)
        problem.set_initial_value(at(crate7, distributor0), True)
        problem.set_initial_value(on(crate7, crate6), True)
        problem.set_initial_value(at(crate8, depot0), True)
        problem.set_initial_value(on(crate8, pallet5), True)
        problem.set_initial_value(at(crate9, distributor0), True)
        problem.set_initial_value(on(crate9, crate7), True)
        problem.set_initial_value(at(crate10, distributor1), True)
        problem.set_initial_value(on(crate10, crate5), True)
        problem.set_initial_value(at(crate11, distributor0), True)
        problem.set_initial_value(on(crate11, pallet4), True)
        problem.set_initial_value(at(crate12, distributor0), True)
        problem.set_initial_value(on(crate12, crate11), True)
        problem.set_initial_value(at(crate13, distributor1), True)
        problem.set_initial_value(on(crate13, crate4), True)
        problem.set_initial_value(at(crate14, distributor0), True)
        problem.set_initial_value(on(crate14, crate9), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate3, pallet0),
            on(crate13, crate3),
            on(crate11, pallet1),
            on(crate10, pallet2),
            on(crate2, crate10),
            on(crate1, crate2),
            on(crate9, crate1),
            on(crate14, pallet3),
            on(crate12, crate14),
            on(crate6, pallet4),
            on(crate4, crate6),
            on(crate5, pallet5),
            on(crate0, crate5),
        )
        problem.set_goal_network(gn)

    if problem_instance == 10:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        depot1 = problem.add_object("depot1", Depot)
        depot2 = problem.add_object("depot2", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        distributor2 = problem.add_object("distributor2", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        pallet3 = problem.add_object("pallet3", Pallet)
        pallet4 = problem.add_object("pallet4", Pallet)
        pallet5 = problem.add_object("pallet5", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)
        hoist3 = problem.add_object("hoist3", Hoist)
        hoist4 = problem.add_object("hoist4", Hoist)
        hoist5 = problem.add_object("hoist5", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate1), True)
        problem.set_initial_value(at(pallet1, depot1), True)
        problem.set_initial_value(clear(crate0), True)
        problem.set_initial_value(at(pallet2, depot2), True)
        problem.set_initial_value(clear(crate4), True)
        problem.set_initial_value(at(pallet3, distributor0), True)
        problem.set_initial_value(clear(crate5), True)
        problem.set_initial_value(at(pallet4, distributor1), True)
        problem.set_initial_value(clear(pallet4), True)
        problem.set_initial_value(at(pallet5, distributor2), True)
        problem.set_initial_value(clear(crate3), True)
        problem.set_initial_value(at(truck0, depot1), True)
        problem.set_initial_value(at(truck1, depot2), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, depot1), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, depot2), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(hoist3, distributor0), True)
        problem.set_initial_value(available(hoist3), True)
        problem.set_initial_value(at(hoist4, distributor1), True)
        problem.set_initial_value(available(hoist4), True)
        problem.set_initial_value(at(hoist5, distributor2), True)
        problem.set_initial_value(available(hoist5), True)
        problem.set_initial_value(at(crate0, depot1), True)
        problem.set_initial_value(on(crate0, pallet1), True)
        problem.set_initial_value(at(crate1, depot0), True)
        problem.set_initial_value(on(crate1, pallet0), True)
        problem.set_initial_value(at(crate2, distributor2), True)
        problem.set_initial_value(on(crate2, pallet5), True)
        problem.set_initial_value(at(crate3, distributor2), True)
        problem.set_initial_value(on(crate3, crate2), True)
        problem.set_initial_value(at(crate4, depot2), True)
        problem.set_initial_value(on(crate4, pallet2), True)
        problem.set_initial_value(at(crate5, distributor0), True)
        problem.set_initial_value(on(crate5, pallet3), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate3, pallet0),
            on(crate2, pallet3),
            on(crate4, pallet5),
            on(crate0, crate4),
        )
        problem.set_goal_network(gn)

    if problem_instance == 11:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        depot1 = problem.add_object("depot1", Depot)
        depot2 = problem.add_object("depot2", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        distributor2 = problem.add_object("distributor2", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        pallet3 = problem.add_object("pallet3", Pallet)
        pallet4 = problem.add_object("pallet4", Pallet)
        pallet5 = problem.add_object("pallet5", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        crate6 = problem.add_object("crate6", Crate)
        crate7 = problem.add_object("crate7", Crate)
        crate8 = problem.add_object("crate8", Crate)
        crate9 = problem.add_object("crate9", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)
        hoist3 = problem.add_object("hoist3", Hoist)
        hoist4 = problem.add_object("hoist4", Hoist)
        hoist5 = problem.add_object("hoist5", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate1), True)
        problem.set_initial_value(at(pallet1, depot1), True)
        problem.set_initial_value(clear(crate3), True)
        problem.set_initial_value(at(pallet2, depot2), True)
        problem.set_initial_value(clear(crate9), True)
        problem.set_initial_value(at(pallet3, distributor0), True)
        problem.set_initial_value(clear(pallet3), True)
        problem.set_initial_value(at(pallet4, distributor1), True)
        problem.set_initial_value(clear(pallet4), True)
        problem.set_initial_value(at(pallet5, distributor2), True)
        problem.set_initial_value(clear(crate8), True)
        problem.set_initial_value(at(truck0, depot2), True)
        problem.set_initial_value(at(truck1, distributor0), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, depot1), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, depot2), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(hoist3, distributor0), True)
        problem.set_initial_value(available(hoist3), True)
        problem.set_initial_value(at(hoist4, distributor1), True)
        problem.set_initial_value(available(hoist4), True)
        problem.set_initial_value(at(hoist5, distributor2), True)
        problem.set_initial_value(available(hoist5), True)
        problem.set_initial_value(at(crate0, depot1), True)
        problem.set_initial_value(on(crate0, pallet1), True)
        problem.set_initial_value(at(crate1, depot0), True)
        problem.set_initial_value(on(crate1, pallet0), True)
        problem.set_initial_value(at(crate2, depot2), True)
        problem.set_initial_value(on(crate2, pallet2), True)
        problem.set_initial_value(at(crate3, depot1), True)
        problem.set_initial_value(on(crate3, crate0), True)
        problem.set_initial_value(at(crate4, depot2), True)
        problem.set_initial_value(on(crate4, crate2), True)
        problem.set_initial_value(at(crate5, depot2), True)
        problem.set_initial_value(on(crate5, crate4), True)
        problem.set_initial_value(at(crate6, distributor2), True)
        problem.set_initial_value(on(crate6, pallet5), True)
        problem.set_initial_value(at(crate7, distributor2), True)
        problem.set_initial_value(on(crate7, crate6), True)
        problem.set_initial_value(at(crate8, distributor2), True)
        problem.set_initial_value(on(crate8, crate7), True)
        problem.set_initial_value(at(crate9, depot2), True)
        problem.set_initial_value(on(crate9, crate5), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate4, pallet0),
            on(crate5, pallet2),
            on(crate6, crate5),
            on(crate8, pallet3),
            on(crate1, pallet4),
            on(crate7, crate1),
            on(crate0, crate7),
            on(crate2, pallet5),
            on(crate9, crate2),
            on(crate3, crate9),
        )
        problem.set_goal_network(gn)

    if problem_instance == 12:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        depot1 = problem.add_object("depot1", Depot)
        depot2 = problem.add_object("depot2", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        distributor2 = problem.add_object("distributor2", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        pallet3 = problem.add_object("pallet3", Pallet)
        pallet4 = problem.add_object("pallet4", Pallet)
        pallet5 = problem.add_object("pallet5", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        crate6 = problem.add_object("crate6", Crate)
        crate7 = problem.add_object("crate7", Crate)
        crate8 = problem.add_object("crate8", Crate)
        crate9 = problem.add_object("crate9", Crate)
        crate10 = problem.add_object("crate10", Crate)
        crate11 = problem.add_object("crate11", Crate)
        crate12 = problem.add_object("crate12", Crate)
        crate13 = problem.add_object("crate13", Crate)
        crate14 = problem.add_object("crate14", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)
        hoist3 = problem.add_object("hoist3", Hoist)
        hoist4 = problem.add_object("hoist4", Hoist)
        hoist5 = problem.add_object("hoist5", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(pallet0), True)
        problem.set_initial_value(at(pallet1, depot1), True)
        problem.set_initial_value(clear(crate12), True)
        problem.set_initial_value(at(pallet2, depot2), True)
        problem.set_initial_value(clear(pallet2), True)
        problem.set_initial_value(at(pallet3, distributor0), True)
        problem.set_initial_value(clear(crate4), True)
        problem.set_initial_value(at(pallet4, distributor1), True)
        problem.set_initial_value(clear(crate14), True)
        problem.set_initial_value(at(pallet5, distributor2), True)
        problem.set_initial_value(clear(crate13), True)
        problem.set_initial_value(at(truck0, distributor1), True)
        problem.set_initial_value(at(truck1, depot1), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, depot1), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, depot2), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(hoist3, distributor0), True)
        problem.set_initial_value(available(hoist3), True)
        problem.set_initial_value(at(hoist4, distributor1), True)
        problem.set_initial_value(available(hoist4), True)
        problem.set_initial_value(at(hoist5, distributor2), True)
        problem.set_initial_value(available(hoist5), True)
        problem.set_initial_value(at(crate0, distributor2), True)
        problem.set_initial_value(on(crate0, pallet5), True)
        problem.set_initial_value(at(crate1, depot1), True)
        problem.set_initial_value(on(crate1, pallet1), True)
        problem.set_initial_value(at(crate2, distributor0), True)
        problem.set_initial_value(on(crate2, pallet3), True)
        problem.set_initial_value(at(crate3, distributor2), True)
        problem.set_initial_value(on(crate3, crate0), True)
        problem.set_initial_value(at(crate4, distributor0), True)
        problem.set_initial_value(on(crate4, crate2), True)
        problem.set_initial_value(at(crate5, depot1), True)
        problem.set_initial_value(on(crate5, crate1), True)
        problem.set_initial_value(at(crate6, distributor2), True)
        problem.set_initial_value(on(crate6, crate3), True)
        problem.set_initial_value(at(crate7, distributor2), True)
        problem.set_initial_value(on(crate7, crate6), True)
        problem.set_initial_value(at(crate8, distributor2), True)
        problem.set_initial_value(on(crate8, crate7), True)
        problem.set_initial_value(at(crate9, distributor2), True)
        problem.set_initial_value(on(crate9, crate8), True)
        problem.set_initial_value(at(crate10, depot1), True)
        problem.set_initial_value(on(crate10, crate5), True)
        problem.set_initial_value(at(crate11, distributor1), True)
        problem.set_initial_value(on(crate11, pallet4), True)
        problem.set_initial_value(at(crate12, depot1), True)
        problem.set_initial_value(on(crate12, crate10), True)
        problem.set_initial_value(at(crate13, distributor2), True)
        problem.set_initial_value(on(crate13, crate9), True)
        problem.set_initial_value(at(crate14, distributor1), True)
        problem.set_initial_value(on(crate14, crate11), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate5, pallet0),
            on(crate13, pallet1),
            on(crate10, crate13),
            on(crate14, crate10),
            on(crate9, pallet2),
            on(crate3, crate9),
            on(crate0, pallet4),
            on(crate2, crate0),
            on(crate6, crate2),
            on(crate12, pallet5),
            on(crate1, crate12),
        )
        problem.set_goal_network(gn)

    if problem_instance == 13:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        depot1 = problem.add_object("depot1", Depot)
        depot2 = problem.add_object("depot2", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        distributor2 = problem.add_object("distributor2", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        pallet3 = problem.add_object("pallet3", Pallet)
        pallet4 = problem.add_object("pallet4", Pallet)
        pallet5 = problem.add_object("pallet5", Pallet)
        pallet6 = problem.add_object("pallet6", Pallet)
        pallet7 = problem.add_object("pallet7", Pallet)
        pallet8 = problem.add_object("pallet8", Pallet)
        pallet9 = problem.add_object("pallet9", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)
        hoist3 = problem.add_object("hoist3", Hoist)
        hoist4 = problem.add_object("hoist4", Hoist)
        hoist5 = problem.add_object("hoist5", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate2), True)
        problem.set_initial_value(at(pallet1, depot1), True)
        problem.set_initial_value(clear(pallet1), True)
        problem.set_initial_value(at(pallet2, depot2), True)
        problem.set_initial_value(clear(crate5), True)
        problem.set_initial_value(at(pallet3, distributor0), True)
        problem.set_initial_value(clear(crate4), True)
        problem.set_initial_value(at(pallet4, distributor1), True)
        problem.set_initial_value(clear(pallet4), True)
        problem.set_initial_value(at(pallet5, distributor2), True)
        problem.set_initial_value(clear(pallet5), True)
        problem.set_initial_value(at(pallet6, distributor1), True)
        problem.set_initial_value(clear(pallet6), True)
        problem.set_initial_value(at(pallet7, depot0), True)
        problem.set_initial_value(clear(pallet7), True)
        problem.set_initial_value(at(pallet8, depot0), True)
        problem.set_initial_value(clear(crate3), True)
        problem.set_initial_value(at(pallet9, distributor0), True)
        problem.set_initial_value(clear(pallet9), True)
        problem.set_initial_value(at(truck0, distributor1), True)
        problem.set_initial_value(at(truck1, depot0), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, depot1), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, depot2), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(hoist3, distributor0), True)
        problem.set_initial_value(available(hoist3), True)
        problem.set_initial_value(at(hoist4, distributor1), True)
        problem.set_initial_value(available(hoist4), True)
        problem.set_initial_value(at(hoist5, distributor2), True)
        problem.set_initial_value(available(hoist5), True)
        problem.set_initial_value(at(crate0, depot2), True)
        problem.set_initial_value(on(crate0, pallet2), True)
        problem.set_initial_value(at(crate1, depot2), True)
        problem.set_initial_value(on(crate1, crate0), True)
        problem.set_initial_value(at(crate2, depot0), True)
        problem.set_initial_value(on(crate2, pallet0), True)
        problem.set_initial_value(at(crate3, depot0), True)
        problem.set_initial_value(on(crate3, pallet8), True)
        problem.set_initial_value(at(crate4, distributor0), True)
        problem.set_initial_value(on(crate4, pallet3), True)
        problem.set_initial_value(at(crate5, depot2), True)
        problem.set_initial_value(on(crate5, crate1), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate0, pallet0),
            on(crate1, pallet5),
            on(crate2, pallet4),
            on(crate3, pallet7),
            on(crate4, pallet9),
            on(crate5, pallet1),
        )
        problem.set_goal_network(gn)

    if problem_instance == 14:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        depot1 = problem.add_object("depot1", Depot)
        depot2 = problem.add_object("depot2", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        distributor2 = problem.add_object("distributor2", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        pallet3 = problem.add_object("pallet3", Pallet)
        pallet4 = problem.add_object("pallet4", Pallet)
        pallet5 = problem.add_object("pallet5", Pallet)
        pallet6 = problem.add_object("pallet6", Pallet)
        pallet7 = problem.add_object("pallet7", Pallet)
        pallet8 = problem.add_object("pallet8", Pallet)
        pallet9 = problem.add_object("pallet9", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        crate6 = problem.add_object("crate6", Crate)
        crate7 = problem.add_object("crate7", Crate)
        crate8 = problem.add_object("crate8", Crate)
        crate9 = problem.add_object("crate9", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)
        hoist3 = problem.add_object("hoist3", Hoist)
        hoist4 = problem.add_object("hoist4", Hoist)
        hoist5 = problem.add_object("hoist5", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(crate4), True)
        problem.set_initial_value(at(pallet1, depot1), True)
        problem.set_initial_value(clear(crate8), True)
        problem.set_initial_value(at(pallet2, depot2), True)
        problem.set_initial_value(clear(pallet2), True)
        problem.set_initial_value(at(pallet3, distributor0), True)
        problem.set_initial_value(clear(crate9), True)
        problem.set_initial_value(at(pallet4, distributor1), True)
        problem.set_initial_value(clear(crate7), True)
        problem.set_initial_value(at(pallet5, distributor2), True)
        problem.set_initial_value(clear(pallet5), True)
        problem.set_initial_value(at(pallet6, distributor2), True)
        problem.set_initial_value(clear(crate3), True)
        problem.set_initial_value(at(pallet7, depot1), True)
        problem.set_initial_value(clear(pallet7), True)
        problem.set_initial_value(at(pallet8, distributor1), True)
        problem.set_initial_value(clear(crate0), True)
        problem.set_initial_value(at(pallet9, depot0), True)
        problem.set_initial_value(clear(crate5), True)
        problem.set_initial_value(at(truck0, depot1), True)
        problem.set_initial_value(at(truck1, depot2), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, depot1), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, depot2), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(hoist3, distributor0), True)
        problem.set_initial_value(available(hoist3), True)
        problem.set_initial_value(at(hoist4, distributor1), True)
        problem.set_initial_value(available(hoist4), True)
        problem.set_initial_value(at(hoist5, distributor2), True)
        problem.set_initial_value(available(hoist5), True)
        problem.set_initial_value(at(crate0, distributor1), True)
        problem.set_initial_value(on(crate0, pallet8), True)
        problem.set_initial_value(at(crate1, depot0), True)
        problem.set_initial_value(on(crate1, pallet9), True)
        problem.set_initial_value(at(crate2, distributor0), True)
        problem.set_initial_value(on(crate2, pallet3), True)
        problem.set_initial_value(at(crate3, distributor2), True)
        problem.set_initial_value(on(crate3, pallet6), True)
        problem.set_initial_value(at(crate4, depot0), True)
        problem.set_initial_value(on(crate4, pallet0), True)
        problem.set_initial_value(at(crate5, depot0), True)
        problem.set_initial_value(on(crate5, crate1), True)
        problem.set_initial_value(at(crate6, distributor1), True)
        problem.set_initial_value(on(crate6, pallet4), True)
        problem.set_initial_value(at(crate7, distributor1), True)
        problem.set_initial_value(on(crate7, crate6), True)
        problem.set_initial_value(at(crate8, depot1), True)
        problem.set_initial_value(on(crate8, pallet1), True)
        problem.set_initial_value(at(crate9, distributor0), True)
        problem.set_initial_value(on(crate9, crate2), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate6, pallet1),
            on(crate7, crate6),
            on(crate9, crate7),
            on(crate5, pallet5),
            on(crate4, pallet0),
            on(crate2, pallet3),
            on(crate1, pallet8),
        )
        problem.set_goal_network(gn)

    if problem_instance == 15:
        # 1. Define Objects
        depot0 = problem.add_object("depot0", Depot)
        depot1 = problem.add_object("depot1", Depot)
        depot2 = problem.add_object("depot2", Depot)
        distributor0 = problem.add_object("distributor0", Distributor)
        distributor1 = problem.add_object("distributor1", Distributor)
        distributor2 = problem.add_object("distributor2", Distributor)
        truck0 = problem.add_object("truck0", Truck)
        truck1 = problem.add_object("truck1", Truck)
        pallet0 = problem.add_object("pallet0", Pallet)
        pallet1 = problem.add_object("pallet1", Pallet)
        pallet2 = problem.add_object("pallet2", Pallet)
        pallet3 = problem.add_object("pallet3", Pallet)
        pallet4 = problem.add_object("pallet4", Pallet)
        pallet5 = problem.add_object("pallet5", Pallet)
        pallet6 = problem.add_object("pallet6", Pallet)
        pallet7 = problem.add_object("pallet7", Pallet)
        pallet8 = problem.add_object("pallet8", Pallet)
        pallet9 = problem.add_object("pallet9", Pallet)
        crate0 = problem.add_object("crate0", Crate)
        crate1 = problem.add_object("crate1", Crate)
        crate2 = problem.add_object("crate2", Crate)
        crate3 = problem.add_object("crate3", Crate)
        crate4 = problem.add_object("crate4", Crate)
        crate5 = problem.add_object("crate5", Crate)
        crate6 = problem.add_object("crate6", Crate)
        crate7 = problem.add_object("crate7", Crate)
        crate8 = problem.add_object("crate8", Crate)
        crate9 = problem.add_object("crate9", Crate)
        crate10 = problem.add_object("crate10", Crate)
        crate11 = problem.add_object("crate11", Crate)
        crate12 = problem.add_object("crate12", Crate)
        crate13 = problem.add_object("crate13", Crate)
        crate14 = problem.add_object("crate14", Crate)
        hoist0 = problem.add_object("hoist0", Hoist)
        hoist1 = problem.add_object("hoist1", Hoist)
        hoist2 = problem.add_object("hoist2", Hoist)
        hoist3 = problem.add_object("hoist3", Hoist)
        hoist4 = problem.add_object("hoist4", Hoist)
        hoist5 = problem.add_object("hoist5", Hoist)

        # 2. Set Initial State
        problem.set_initial_value(at(pallet0, depot0), True)
        problem.set_initial_value(clear(pallet0), True)
        problem.set_initial_value(at(pallet1, depot1), True)
        problem.set_initial_value(clear(crate7), True)
        problem.set_initial_value(at(pallet2, depot2), True)
        problem.set_initial_value(clear(pallet2), True)
        problem.set_initial_value(at(pallet3, distributor0), True)
        problem.set_initial_value(clear(crate8), True)
        problem.set_initial_value(at(pallet4, distributor1), True)
        problem.set_initial_value(clear(crate12), True)
        problem.set_initial_value(at(pallet5, distributor2), True)
        problem.set_initial_value(clear(crate11), True)
        problem.set_initial_value(at(pallet6, depot1), True)
        problem.set_initial_value(clear(crate4), True)
        problem.set_initial_value(at(pallet7, distributor0), True)
        problem.set_initial_value(clear(crate9), True)
        problem.set_initial_value(at(pallet8, depot2), True)
        problem.set_initial_value(clear(crate13), True)
        problem.set_initial_value(at(pallet9, distributor0), True)
        problem.set_initial_value(clear(crate14), True)
        problem.set_initial_value(at(truck0, distributor1), True)
        problem.set_initial_value(at(truck1, distributor2), True)
        problem.set_initial_value(at(hoist0, depot0), True)
        problem.set_initial_value(available(hoist0), True)
        problem.set_initial_value(at(hoist1, depot1), True)
        problem.set_initial_value(available(hoist1), True)
        problem.set_initial_value(at(hoist2, depot2), True)
        problem.set_initial_value(available(hoist2), True)
        problem.set_initial_value(at(hoist3, distributor0), True)
        problem.set_initial_value(available(hoist3), True)
        problem.set_initial_value(at(hoist4, distributor1), True)
        problem.set_initial_value(available(hoist4), True)
        problem.set_initial_value(at(hoist5, distributor2), True)
        problem.set_initial_value(available(hoist5), True)
        problem.set_initial_value(at(crate0, distributor2), True)
        problem.set_initial_value(on(crate0, pallet5), True)
        problem.set_initial_value(at(crate1, distributor1), True)
        problem.set_initial_value(on(crate1, pallet4), True)
        problem.set_initial_value(at(crate2, depot2), True)
        problem.set_initial_value(on(crate2, pallet8), True)
        problem.set_initial_value(at(crate3, depot2), True)
        problem.set_initial_value(on(crate3, crate2), True)
        problem.set_initial_value(at(crate4, depot1), True)
        problem.set_initial_value(on(crate4, pallet6), True)
        problem.set_initial_value(at(crate5, distributor2), True)
        problem.set_initial_value(on(crate5, crate0), True)
        problem.set_initial_value(at(crate6, depot1), True)
        problem.set_initial_value(on(crate6, pallet1), True)
        problem.set_initial_value(at(crate7, depot1), True)
        problem.set_initial_value(on(crate7, crate6), True)
        problem.set_initial_value(at(crate8, distributor0), True)
        problem.set_initial_value(on(crate8, pallet3), True)
        problem.set_initial_value(at(crate9, distributor0), True)
        problem.set_initial_value(on(crate9, pallet7), True)
        problem.set_initial_value(at(crate10, distributor1), True)
        problem.set_initial_value(on(crate10, crate1), True)
        problem.set_initial_value(at(crate11, distributor2), True)
        problem.set_initial_value(on(crate11, crate5), True)
        problem.set_initial_value(at(crate12, distributor1), True)
        problem.set_initial_value(on(crate12, crate10), True)
        problem.set_initial_value(at(crate13, depot2), True)
        problem.set_initial_value(on(crate13, crate3), True)
        problem.set_initial_value(at(crate14, distributor0), True)
        problem.set_initial_value(on(crate14, pallet9), True)

        # 3. Set Goal Network (HTN)
        # The do_put_on tasks are translated to 'on' predicates as the desired state.
        gn = PartialOrderGoalNetwork()
        gn.add(
            on(crate2, pallet0),
            on(crate3, pallet1),
            on(crate7, pallet4),
            on(crate4, crate7),
            on(crate9, crate4),
            on(crate11, crate9),
            on(crate10, crate11),
            on(crate1, crate10),
            on(crate5, pallet5),
            on(crate12, crate5),
            on(crate6, pallet6),
            on(crate8, pallet7),
            on(crate0, crate8),
            on(crate13, pallet8),
            on(crate14, pallet9),
        )
        problem.set_goal_network(gn)
    return problem
