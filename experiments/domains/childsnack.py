from unified_planning.shortcuts import *
from unified_planning.model.phgn import *
from unified_planning.model.phgn.goal_network import PartialOrderGoalNetwork


def childsnack(problem_instance: int = 1):
    problem = PHGNProblem("child_snack")

    # 1. Define Types
    Child = UserType("Child")
    BreadPortion = UserType("BreadPortion")
    ContentPortion = UserType("ContentPortion")
    Sandwich = UserType("Sandwich")
    Tray = UserType("Tray")
    Place = UserType("Place")

    # 2. Define Constants
    kitchen = problem.add_object("kitchen", Place)

    # 3. Define Fluents (Predicates)
    at_kitchen_bread = problem.add_fluent(
        "at_kitchen_bread", BoolType(), default_initial_value=False, b=BreadPortion
    )
    at_kitchen_content = problem.add_fluent(
        "at_kitchen_content", BoolType(), default_initial_value=False, c=ContentPortion
    )
    at_kitchen_sandwich = problem.add_fluent(
        "at_kitchen_sandwich", BoolType(), default_initial_value=False, s=Sandwich
    )
    no_gluten_bread = problem.add_fluent(
        "no_gluten_bread", BoolType(), default_initial_value=False, b=BreadPortion
    )
    no_gluten_content = problem.add_fluent(
        "no_gluten_content", BoolType(), default_initial_value=False, c=ContentPortion
    )
    ontray = problem.add_fluent(
        "ontray", BoolType(), default_initial_value=False, s=Sandwich, t=Tray
    )
    no_gluten_sandwich = problem.add_fluent(
        "no_gluten_sandwich", BoolType(), default_initial_value=False, s=Sandwich
    )
    allergic_gluten = problem.add_fluent(
        "allergic_gluten", BoolType(), default_initial_value=False, c=Child
    )
    not_allergic_gluten = problem.add_fluent(
        "not_allergic_gluten", BoolType(), default_initial_value=False, c=Child
    )
    served = problem.add_fluent(
        "served", BoolType(), default_initial_value=False, c=Child
    )
    waiting = problem.add_fluent(
        "waiting", BoolType(), default_initial_value=False, c=Child, p=Place
    )
    at = problem.add_fluent(
        "at", BoolType(), default_initial_value=False, t=Tray, p=Place
    )
    notexist = problem.add_fluent(
        "notexist", BoolType(), default_initial_value=False, s=Sandwich
    )
    dirty_tray = problem.add_fluent(
        "dirty_tray", BoolType(), default_initial_value=False, t=Tray
    )

    # 4. Define Actions

    # make_sandwich_no_gluten
    make_sandwich_no_gluten = InstantaneousAction(
        "make_sandwich_no_gluten", s=Sandwich, b=BreadPortion, c=ContentPortion
    )
    make_sandwich_no_gluten.add_precondition(
        at_kitchen_bread(make_sandwich_no_gluten.b)
    )
    make_sandwich_no_gluten.add_precondition(
        at_kitchen_content(make_sandwich_no_gluten.c)
    )
    make_sandwich_no_gluten.add_precondition(no_gluten_bread(make_sandwich_no_gluten.b))
    make_sandwich_no_gluten.add_precondition(
        no_gluten_content(make_sandwich_no_gluten.c)
    )
    make_sandwich_no_gluten.add_precondition(notexist(make_sandwich_no_gluten.s))
    make_sandwich_no_gluten.add_effect(
        at_kitchen_bread(make_sandwich_no_gluten.b), False
    )
    make_sandwich_no_gluten.add_effect(
        at_kitchen_content(make_sandwich_no_gluten.c), False
    )
    make_sandwich_no_gluten.add_effect(
        at_kitchen_sandwich(make_sandwich_no_gluten.s), True
    )
    make_sandwich_no_gluten.add_effect(
        no_gluten_sandwich(make_sandwich_no_gluten.s), True
    )
    make_sandwich_no_gluten.add_effect(notexist(make_sandwich_no_gluten.s), False)
    problem.add_action(make_sandwich_no_gluten)

    # make_sandwich
    make_sandwich = InstantaneousAction(
        "make_sandwich", s=Sandwich, b=BreadPortion, c=ContentPortion
    )
    make_sandwich.add_precondition(at_kitchen_bread(make_sandwich.b))
    make_sandwich.add_precondition(at_kitchen_content(make_sandwich.c))
    make_sandwich.add_precondition(notexist(make_sandwich.s))
    make_sandwich.add_effect(at_kitchen_bread(make_sandwich.b), False)
    make_sandwich.add_effect(at_kitchen_content(make_sandwich.c), False)
    make_sandwich.add_effect(at_kitchen_sandwich(make_sandwich.s), True)
    make_sandwich.add_effect(notexist(make_sandwich.s), False)
    problem.add_action(make_sandwich)

    # put_on_tray
    put_on_tray = InstantaneousAction("put_on_tray", s=Sandwich, t=Tray)
    put_on_tray.add_precondition(at_kitchen_sandwich(put_on_tray.s))
    put_on_tray.add_precondition(at(put_on_tray.t, kitchen))
    put_on_tray.add_precondition(Not(dirty_tray(put_on_tray.t)))
    put_on_tray.add_effect(at_kitchen_sandwich(put_on_tray.s), False)
    put_on_tray.add_effect(ontray(put_on_tray.s, put_on_tray.t), True)
    problem.add_action(put_on_tray)

    # serve_sandwich_no_gluten
    serve_sandwich_no_gluten = ProbabilisticAction(
        "serve_sandwich_no_gluten", s=Sandwich, c=Child, t=Tray, p=Place
    )
    serve_sandwich_no_gluten.add_precondition(
        allergic_gluten(serve_sandwich_no_gluten.c)
    )
    serve_sandwich_no_gluten.add_precondition(
        ontray(serve_sandwich_no_gluten.s, serve_sandwich_no_gluten.t)
    )
    serve_sandwich_no_gluten.add_precondition(
        waiting(serve_sandwich_no_gluten.c, serve_sandwich_no_gluten.p)
    )
    serve_sandwich_no_gluten.add_precondition(
        no_gluten_sandwich(serve_sandwich_no_gluten.s)
    )
    serve_sandwich_no_gluten.add_precondition(
        at(serve_sandwich_no_gluten.t, serve_sandwich_no_gluten.p)
    )

    serve_sandwich_no_gluten.add_outcome("normal", 0.5)
    serve_sandwich_no_gluten.add_effect(
        "normal", ontray(serve_sandwich_no_gluten.s, serve_sandwich_no_gluten.t), False
    )
    serve_sandwich_no_gluten.add_effect(
        "normal", served(serve_sandwich_no_gluten.c), True
    )

    serve_sandwich_no_gluten.add_outcome("dirty", 0.5)
    serve_sandwich_no_gluten.add_effect(
        "dirty", ontray(serve_sandwich_no_gluten.s, serve_sandwich_no_gluten.t), False
    )
    serve_sandwich_no_gluten.add_effect(
        "dirty", served(serve_sandwich_no_gluten.c), True
    )
    serve_sandwich_no_gluten.add_effect(
        "dirty", dirty_tray(serve_sandwich_no_gluten.t), True
    )
    problem.add_action(serve_sandwich_no_gluten)

    # serve_sandwich
    serve_sandwich = ProbabilisticAction(
        "serve_sandwich", s=Sandwich, c=Child, t=Tray, p=Place
    )
    serve_sandwich.add_precondition(not_allergic_gluten(serve_sandwich.c))
    serve_sandwich.add_precondition(waiting(serve_sandwich.c, serve_sandwich.p))
    serve_sandwich.add_precondition(ontray(serve_sandwich.s, serve_sandwich.t))
    serve_sandwich.add_precondition(at(serve_sandwich.t, serve_sandwich.p))

    serve_sandwich.add_outcome("normal", 0.5)
    serve_sandwich.add_effect(
        "normal", ontray(serve_sandwich.s, serve_sandwich.t), False
    )
    serve_sandwich.add_effect("normal", served(serve_sandwich.c), True)

    serve_sandwich.add_outcome("dirty", 0.5)
    serve_sandwich.add_effect(
        "dirty", ontray(serve_sandwich.s, serve_sandwich.t), False
    )
    serve_sandwich.add_effect("dirty", served(serve_sandwich.c), True)
    serve_sandwich.add_effect("dirty", dirty_tray(serve_sandwich.t), True)
    problem.add_action(serve_sandwich)

    # move_tray
    move_tray = InstantaneousAction("move_tray", t=Tray, p1=Place, p2=Place)
    move_tray.add_precondition(at(move_tray.t, move_tray.p1))
    move_tray.add_effect(at(move_tray.t, move_tray.p1), False)
    move_tray.add_effect(at(move_tray.t, move_tray.p2), True)
    problem.add_action(move_tray)

    # wash_tray_in_kitchen
    wash_tray_in_kitchen = InstantaneousAction("wash_tray_in_kitchen", t=Tray, p=Place)
    wash_tray_in_kitchen.add_precondition(
        at(wash_tray_in_kitchen.t, wash_tray_in_kitchen.p)
    )
    wash_tray_in_kitchen.add_precondition(dirty_tray(wash_tray_in_kitchen.t))
    wash_tray_in_kitchen.add_effect(dirty_tray(wash_tray_in_kitchen.t), False)
    problem.add_action(wash_tray_in_kitchen)

    serve_0 = PHGNMethod(
        "serve_0",
        c=Child,
        s=Sandwich,
        t=Tray,
        p2=Place,
    )
    serve_0.add_precondition(allergic_gluten(serve_0.c))
    serve_0.add_precondition(notexist(serve_0.s))
    serve_0.add_precondition(waiting(serve_0.c, serve_0.p2))
    gn = PartialOrderGoalNetwork()
    gn.add(
        no_gluten_sandwich(serve_0.s),
        at(serve_0.t, kitchen),
        Not(dirty_tray(serve_0.t)),
        ontray(serve_0.s, serve_0.t),
        at(serve_0.t, serve_0.p2),
        served(serve_0.c),
    )
    serve_0.set_goal_network(gn)
    problem.add_method(serve_0)

    serve_1 = PHGNMethod(
        "serve_1",
        c=Child,
        s=Sandwich,
        t=Tray,
        p2=Place,
    )
    serve_1.add_precondition(not_allergic_gluten(serve_1.c))
    serve_1.add_precondition(notexist(serve_1.s))
    serve_1.add_precondition(waiting(serve_1.c, serve_1.p2))
    gn = PartialOrderGoalNetwork()
    gn.add(
        Not(notexist(serve_1.s)),
        at(serve_1.t, kitchen),
        Not(dirty_tray(serve_1.t)),
        ontray(serve_1.s, serve_1.t),
        at(serve_1.t, serve_1.p2),
        served(serve_1.c),
    )
    serve_1.set_goal_network(gn)
    problem.add_method(serve_1)

    if problem_instance == 1:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)

        problem.set_initial_value(no_gluten_bread(bread2), True)
        problem.set_initial_value(no_gluten_content(content2), True)
        problem.set_initial_value(no_gluten_content(content8), True)
        problem.set_initial_value(no_gluten_content(content4), True)
        problem.set_initial_value(no_gluten_content(content1), True)

        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(not_allergic_gluten(child2), True)
        problem.set_initial_value(not_allergic_gluten(child3), True)

        problem.set_initial_value(waiting(child1, table2), True)
        problem.set_initial_value(waiting(child2, table1), True)
        problem.set_initial_value(waiting(child3, table1), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(served(child1), served(child2), served(child3))
        problem.set_goal_network(gn)

    if problem_instance == 2:
        # 1. Define Objects (Instances of UserTypes)
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)

        # 2. Set Initial State
        # Fluents are False by default, so we only need to set the ones that are True.
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)  # New bread

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)

        problem.set_initial_value(no_gluten_bread(bread2), True)
        problem.set_initial_value(no_gluten_content(content2), True)
        problem.set_initial_value(no_gluten_content(content5), True)
        problem.set_initial_value(no_gluten_content(content6), True)

        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(not_allergic_gluten(child2), True)
        problem.set_initial_value(not_allergic_gluten(child3), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)

        problem.set_initial_value(waiting(child1, table2), True)
        problem.set_initial_value(waiting(child2, table1), True)
        problem.set_initial_value(waiting(child3, table1), True)
        problem.set_initial_value(waiting(child4, table2), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(served(child1), served(child2), served(child3), served(child4))
        problem.set_goal_network(gn)

    if problem_instance == 3:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)

        problem.set_initial_value(no_gluten_bread(bread2), True)
        problem.set_initial_value(no_gluten_content(content2), True)
        problem.set_initial_value(no_gluten_content(content6), True)

        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(not_allergic_gluten(child2), True)
        problem.set_initial_value(not_allergic_gluten(child3), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)
        problem.set_initial_value(not_allergic_gluten(child5), True)

        problem.set_initial_value(waiting(child1, table2), True)
        problem.set_initial_value(waiting(child2, table1), True)
        problem.set_initial_value(waiting(child3, table1), True)
        problem.set_initial_value(waiting(child4, table2), True)
        problem.set_initial_value(waiting(child5, table3), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
        )
        problem.set_goal_network(gn)

    if problem_instance == 4:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)

        problem.set_initial_value(no_gluten_bread(bread3), True)
        problem.set_initial_value(no_gluten_bread(bread11), True)
        problem.set_initial_value(no_gluten_bread(bread4), True)
        problem.set_initial_value(no_gluten_bread(bread5), True)
        problem.set_initial_value(no_gluten_content(content2), True)
        problem.set_initial_value(no_gluten_content(content9), True)
        problem.set_initial_value(no_gluten_content(content5), True)
        problem.set_initial_value(no_gluten_content(content12), True)

        problem.set_initial_value(allergic_gluten(child12), True)
        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(allergic_gluten(child3), True)
        problem.set_initial_value(allergic_gluten(child5), True)
        problem.set_initial_value(not_allergic_gluten(child2), True)
        problem.set_initial_value(not_allergic_gluten(child11), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)
        problem.set_initial_value(not_allergic_gluten(child6), True)
        problem.set_initial_value(not_allergic_gluten(child7), True)
        problem.set_initial_value(not_allergic_gluten(child8), True)
        problem.set_initial_value(not_allergic_gluten(child9), True)
        problem.set_initial_value(not_allergic_gluten(child10), True)

        problem.set_initial_value(waiting(child1, table2), True)
        problem.set_initial_value(waiting(child2, table1), True)
        problem.set_initial_value(waiting(child3, table1), True)
        problem.set_initial_value(waiting(child4, table2), True)
        problem.set_initial_value(waiting(child5, table3), True)
        problem.set_initial_value(waiting(child6, table3), True)
        problem.set_initial_value(waiting(child7, table3), True)
        problem.set_initial_value(waiting(child8, table2), True)
        problem.set_initial_value(waiting(child9, table1), True)
        problem.set_initial_value(waiting(child10, table3), True)
        problem.set_initial_value(waiting(child11, table1), True)
        problem.set_initial_value(waiting(child12, table1), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
        )
        problem.set_goal_network(gn)

    if problem_instance == 5:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)

        problem.set_initial_value(no_gluten_bread(bread3), True)
        problem.set_initial_value(no_gluten_bread(bread12), True)
        problem.set_initial_value(no_gluten_bread(bread5), True)
        problem.set_initial_value(no_gluten_bread(bread11), True)
        problem.set_initial_value(no_gluten_bread(bread1), True)
        problem.set_initial_value(no_gluten_content(content11), True)
        problem.set_initial_value(no_gluten_content(content6), True)
        problem.set_initial_value(no_gluten_content(content2), True)
        problem.set_initial_value(no_gluten_content(content10), True)
        problem.set_initial_value(no_gluten_content(content4), True)

        problem.set_initial_value(allergic_gluten(child8), True)
        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(allergic_gluten(child12), True)
        problem.set_initial_value(allergic_gluten(child4), True)
        problem.set_initial_value(allergic_gluten(child13), True)
        problem.set_initial_value(not_allergic_gluten(child2), True)
        problem.set_initial_value(not_allergic_gluten(child3), True)
        problem.set_initial_value(not_allergic_gluten(child5), True)
        problem.set_initial_value(not_allergic_gluten(child6), True)
        problem.set_initial_value(not_allergic_gluten(child7), True)
        problem.set_initial_value(not_allergic_gluten(child9), True)
        problem.set_initial_value(not_allergic_gluten(child10), True)
        problem.set_initial_value(not_allergic_gluten(child11), True)

        problem.set_initial_value(waiting(child1, table2), True)
        problem.set_initial_value(waiting(child2, table3), True)
        problem.set_initial_value(waiting(child3, table3), True)
        problem.set_initial_value(waiting(child4, table3), True)
        problem.set_initial_value(waiting(child5, table2), True)
        problem.set_initial_value(waiting(child6, table1), True)
        problem.set_initial_value(waiting(child7, table3), True)
        problem.set_initial_value(waiting(child8, table1), True)
        problem.set_initial_value(waiting(child9, table1), True)
        problem.set_initial_value(waiting(child10, table3), True)
        problem.set_initial_value(waiting(child11, table1), True)
        problem.set_initial_value(waiting(child12, table1), True)
        problem.set_initial_value(waiting(child13, table1), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
        )
        problem.set_goal_network(gn)

    if problem_instance == 6:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)

        problem.set_initial_value(no_gluten_bread(bread8), True)
        problem.set_initial_value(no_gluten_bread(bread3), True)
        problem.set_initial_value(no_gluten_bread(bread11), True)
        problem.set_initial_value(no_gluten_bread(bread10), True)
        problem.set_initial_value(no_gluten_bread(bread5), True)
        problem.set_initial_value(no_gluten_content(content9), True)
        problem.set_initial_value(no_gluten_content(content3), True)
        problem.set_initial_value(no_gluten_content(content12), True)
        problem.set_initial_value(no_gluten_content(content11), True)
        problem.set_initial_value(no_gluten_content(content7), True)

        problem.set_initial_value(allergic_gluten(child12), True)
        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(allergic_gluten(child2), True)
        problem.set_initial_value(allergic_gluten(child3), True)
        problem.set_initial_value(allergic_gluten(child11), True)
        problem.set_initial_value(not_allergic_gluten(child13), True)
        problem.set_initial_value(not_allergic_gluten(child10), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)
        problem.set_initial_value(not_allergic_gluten(child5), True)
        problem.set_initial_value(not_allergic_gluten(child6), True)
        problem.set_initial_value(not_allergic_gluten(child7), True)
        problem.set_initial_value(not_allergic_gluten(child8), True)
        problem.set_initial_value(not_allergic_gluten(child9), True)

        problem.set_initial_value(waiting(child1, table3), True)
        problem.set_initial_value(waiting(child2, table2), True)
        problem.set_initial_value(waiting(child3, table2), True)
        problem.set_initial_value(waiting(child4, table1), True)
        problem.set_initial_value(waiting(child5, table2), True)
        problem.set_initial_value(waiting(child6, table2), True)
        problem.set_initial_value(waiting(child7, table1), True)
        problem.set_initial_value(waiting(child8, table2), True)
        problem.set_initial_value(waiting(child9, table1), True)
        problem.set_initial_value(waiting(child10, table1), True)
        problem.set_initial_value(waiting(child11, table3), True)
        problem.set_initial_value(waiting(child12, table3), True)
        problem.set_initial_value(waiting(child13, table2), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
        )
        problem.set_goal_network(gn)

    if problem_instance == 7:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)
        child14 = problem.add_object("child14", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)
        bread14 = problem.add_object("bread14", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)
        content14 = problem.add_object("content14", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)
        sandw18 = problem.add_object("sandw18", Sandwich)
        sandw19 = problem.add_object("sandw19", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)
        problem.set_initial_value(at_kitchen_bread(bread14), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)
        problem.set_initial_value(at_kitchen_content(content14), True)

        problem.set_initial_value(no_gluten_bread(bread3), True)
        problem.set_initial_value(no_gluten_bread(bread13), True)
        problem.set_initial_value(no_gluten_bread(bread5), True)
        problem.set_initial_value(no_gluten_bread(bread6), True)
        problem.set_initial_value(no_gluten_bread(bread2), True)
        problem.set_initial_value(no_gluten_content(content12), True)
        problem.set_initial_value(no_gluten_content(content7), True)
        problem.set_initial_value(no_gluten_content(content2), True)
        problem.set_initial_value(no_gluten_content(content11), True)
        problem.set_initial_value(no_gluten_content(content5), True)

        problem.set_initial_value(allergic_gluten(child8), True)
        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(allergic_gluten(child4), True)
        problem.set_initial_value(allergic_gluten(child5), True)
        problem.set_initial_value(allergic_gluten(child14), True)
        problem.set_initial_value(not_allergic_gluten(child12), True)
        problem.set_initial_value(not_allergic_gluten(child13), True)
        problem.set_initial_value(not_allergic_gluten(child2), True)
        problem.set_initial_value(not_allergic_gluten(child3), True)
        problem.set_initial_value(not_allergic_gluten(child6), True)
        problem.set_initial_value(not_allergic_gluten(child7), True)
        problem.set_initial_value(not_allergic_gluten(child9), True)
        problem.set_initial_value(not_allergic_gluten(child10), True)
        problem.set_initial_value(not_allergic_gluten(child11), True)

        problem.set_initial_value(waiting(child1, table2), True)
        problem.set_initial_value(waiting(child2, table3), True)
        problem.set_initial_value(waiting(child3, table3), True)
        problem.set_initial_value(waiting(child4, table3), True)
        problem.set_initial_value(waiting(child5, table2), True)
        problem.set_initial_value(waiting(child6, table1), True)
        problem.set_initial_value(waiting(child7, table3), True)
        problem.set_initial_value(waiting(child8, table1), True)
        problem.set_initial_value(waiting(child9, table1), True)
        problem.set_initial_value(waiting(child10, table3), True)
        problem.set_initial_value(waiting(child11, table1), True)
        problem.set_initial_value(waiting(child12, table1), True)
        problem.set_initial_value(waiting(child13, table1), True)
        problem.set_initial_value(waiting(child14, table2), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)
        problem.set_initial_value(notexist(sandw18), True)
        problem.set_initial_value(notexist(sandw19), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
            served(child14),
        )
        problem.set_goal_network(gn)

    if problem_instance == 8:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)
        child14 = problem.add_object("child14", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)
        bread14 = problem.add_object("bread14", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)
        content14 = problem.add_object("content14", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)
        sandw18 = problem.add_object("sandw18", Sandwich)
        sandw19 = problem.add_object("sandw19", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)
        problem.set_initial_value(at_kitchen_bread(bread14), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)
        problem.set_initial_value(at_kitchen_content(content14), True)

        problem.set_initial_value(no_gluten_bread(bread9), True)
        problem.set_initial_value(no_gluten_bread(bread3), True)
        problem.set_initial_value(no_gluten_bread(bread12), True)
        problem.set_initial_value(no_gluten_bread(bread11), True)
        problem.set_initial_value(no_gluten_bread(bread5), True)
        problem.set_initial_value(no_gluten_content(content10), True)
        problem.set_initial_value(no_gluten_content(content3), True)
        problem.set_initial_value(no_gluten_content(content13), True)
        problem.set_initial_value(no_gluten_content(content4), True)
        problem.set_initial_value(no_gluten_content(content7), True)

        problem.set_initial_value(allergic_gluten(child12), True)
        problem.set_initial_value(allergic_gluten(child13), True)
        problem.set_initial_value(allergic_gluten(child3), True)
        problem.set_initial_value(allergic_gluten(child14), True)
        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(not_allergic_gluten(child2), True)
        problem.set_initial_value(not_allergic_gluten(child11), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)
        problem.set_initial_value(not_allergic_gluten(child5), True)
        problem.set_initial_value(not_allergic_gluten(child6), True)
        problem.set_initial_value(not_allergic_gluten(child7), True)
        problem.set_initial_value(not_allergic_gluten(child8), True)
        problem.set_initial_value(not_allergic_gluten(child9), True)
        problem.set_initial_value(not_allergic_gluten(child10), True)

        problem.set_initial_value(waiting(child1, table3), True)
        problem.set_initial_value(waiting(child2, table2), True)
        problem.set_initial_value(waiting(child3, table2), True)
        problem.set_initial_value(waiting(child4, table1), True)
        problem.set_initial_value(waiting(child5, table2), True)
        problem.set_initial_value(waiting(child6, table2), True)
        problem.set_initial_value(waiting(child7, table1), True)
        problem.set_initial_value(waiting(child8, table2), True)
        problem.set_initial_value(waiting(child9, table1), True)
        problem.set_initial_value(waiting(child10, table1), True)
        problem.set_initial_value(waiting(child11, table3), True)
        problem.set_initial_value(waiting(child12, table3), True)
        problem.set_initial_value(waiting(child13, table2), True)
        problem.set_initial_value(waiting(child14, table3), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)
        problem.set_initial_value(notexist(sandw18), True)
        problem.set_initial_value(notexist(sandw19), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
            served(child14),
        )
        problem.set_goal_network(gn)

    if problem_instance == 9:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)
        child14 = problem.add_object("child14", Child)
        child15 = problem.add_object("child15", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)
        bread14 = problem.add_object("bread14", BreadPortion)
        bread15 = problem.add_object("bread15", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)
        content14 = problem.add_object("content14", ContentPortion)
        content15 = problem.add_object("content15", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)
        sandw18 = problem.add_object("sandw18", Sandwich)
        sandw19 = problem.add_object("sandw19", Sandwich)
        sandw20 = problem.add_object("sandw20", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)
        problem.set_initial_value(at_kitchen_bread(bread14), True)
        problem.set_initial_value(at_kitchen_bread(bread15), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)
        problem.set_initial_value(at_kitchen_content(content14), True)
        problem.set_initial_value(at_kitchen_content(content15), True)

        problem.set_initial_value(no_gluten_bread(bread3), True)
        problem.set_initial_value(no_gluten_bread(bread14), True)
        problem.set_initial_value(no_gluten_bread(bread6), True)
        problem.set_initial_value(no_gluten_bread(bread13), True)
        problem.set_initial_value(no_gluten_bread(bread2), True)
        problem.set_initial_value(no_gluten_bread(bread9), True)
        problem.set_initial_value(no_gluten_content(content8), True)
        problem.set_initial_value(no_gluten_content(content2), True)
        problem.set_initial_value(no_gluten_content(content13), True)
        problem.set_initial_value(no_gluten_content(content5), True)
        problem.set_initial_value(no_gluten_content(content1), True)
        problem.set_initial_value(no_gluten_content(content4), True)

        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(allergic_gluten(child10), True)
        problem.set_initial_value(allergic_gluten(child5), True)
        problem.set_initial_value(allergic_gluten(child7), True)
        problem.set_initial_value(allergic_gluten(child8), True)
        problem.set_initial_value(allergic_gluten(child9), True)
        problem.set_initial_value(not_allergic_gluten(child12), True)
        problem.set_initial_value(not_allergic_gluten(child13), True)
        problem.set_initial_value(not_allergic_gluten(child2), True)
        problem.set_initial_value(not_allergic_gluten(child3), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)
        problem.set_initial_value(not_allergic_gluten(child14), True)
        problem.set_initial_value(not_allergic_gluten(child6), True)
        problem.set_initial_value(not_allergic_gluten(child15), True)
        problem.set_initial_value(not_allergic_gluten(child11), True)

        problem.set_initial_value(waiting(child1, table3), True)
        problem.set_initial_value(waiting(child2, table2), True)
        problem.set_initial_value(waiting(child3, table1), True)
        problem.set_initial_value(waiting(child4, table3), True)
        problem.set_initial_value(waiting(child5, table1), True)
        problem.set_initial_value(waiting(child6, table1), True)
        problem.set_initial_value(waiting(child7, table3), True)
        problem.set_initial_value(waiting(child8, table1), True)
        problem.set_initial_value(waiting(child9, table1), True)
        problem.set_initial_value(waiting(child10, table1), True)
        problem.set_initial_value(waiting(child11, table2), True)
        problem.set_initial_value(waiting(child12, table3), True)
        problem.set_initial_value(waiting(child13, table1), True)
        problem.set_initial_value(waiting(child14, table2), True)
        problem.set_initial_value(waiting(child15, table2), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)
        problem.set_initial_value(notexist(sandw18), True)
        problem.set_initial_value(notexist(sandw19), True)
        problem.set_initial_value(notexist(sandw20), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
            served(child14),
            served(child15),
        )
        problem.set_goal_network(gn)

    if problem_instance == 10:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)
        child14 = problem.add_object("child14", Child)
        child15 = problem.add_object("child15", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)
        bread14 = problem.add_object("bread14", BreadPortion)
        bread15 = problem.add_object("bread15", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)
        content14 = problem.add_object("content14", ContentPortion)
        content15 = problem.add_object("content15", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)
        sandw18 = problem.add_object("sandw18", Sandwich)
        sandw19 = problem.add_object("sandw19", Sandwich)
        sandw20 = problem.add_object("sandw20", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)
        problem.set_initial_value(at_kitchen_bread(bread14), True)
        problem.set_initial_value(at_kitchen_bread(bread15), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)
        problem.set_initial_value(at_kitchen_content(content14), True)
        problem.set_initial_value(at_kitchen_content(content15), True)

        problem.set_initial_value(no_gluten_bread(bread9), True)
        problem.set_initial_value(no_gluten_bread(bread3), True)
        problem.set_initial_value(no_gluten_bread(bread13), True)
        problem.set_initial_value(no_gluten_bread(bread12), True)
        problem.set_initial_value(no_gluten_bread(bread6), True)
        problem.set_initial_value(no_gluten_bread(bread7), True)
        problem.set_initial_value(no_gluten_content(content4), True)
        problem.set_initial_value(no_gluten_content(content15), True)
        problem.set_initial_value(no_gluten_content(content14), True)
        problem.set_initial_value(no_gluten_content(content9), True)
        problem.set_initial_value(no_gluten_content(content3), True)
        problem.set_initial_value(no_gluten_content(content10), True)

        problem.set_initial_value(allergic_gluten(child2), True)
        problem.set_initial_value(allergic_gluten(child11), True)
        problem.set_initial_value(allergic_gluten(child5), True)
        problem.set_initial_value(allergic_gluten(child15), True)
        problem.set_initial_value(allergic_gluten(child10), True)
        problem.set_initial_value(allergic_gluten(child3), True)
        problem.set_initial_value(not_allergic_gluten(child12), True)
        problem.set_initial_value(not_allergic_gluten(child1), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)
        problem.set_initial_value(not_allergic_gluten(child14), True)
        problem.set_initial_value(not_allergic_gluten(child6), True)
        problem.set_initial_value(not_allergic_gluten(child7), True)
        problem.set_initial_value(not_allergic_gluten(child8), True)
        problem.set_initial_value(not_allergic_gluten(child9), True)
        problem.set_initial_value(not_allergic_gluten(child13), True)

        problem.set_initial_value(waiting(child1, table1), True)
        problem.set_initial_value(waiting(child2, table2), True)
        problem.set_initial_value(waiting(child3, table2), True)
        problem.set_initial_value(waiting(child4, table1), True)
        problem.set_initial_value(waiting(child5, table2), True)
        problem.set_initial_value(waiting(child6, table1), True)
        problem.set_initial_value(waiting(child7, table1), True)
        problem.set_initial_value(waiting(child8, table3), True)
        problem.set_initial_value(waiting(child9, table3), True)
        problem.set_initial_value(waiting(child10, table2), True)
        problem.set_initial_value(waiting(child11, table3), True)
        problem.set_initial_value(waiting(child12, table2), True)
        problem.set_initial_value(waiting(child13, table2), True)
        problem.set_initial_value(waiting(child14, table2), True)
        problem.set_initial_value(waiting(child15, table2), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)
        problem.set_initial_value(notexist(sandw18), True)
        problem.set_initial_value(notexist(sandw19), True)
        problem.set_initial_value(notexist(sandw20), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
            served(child14),
            served(child15),
        )
        problem.set_goal_network(gn)

    if problem_instance == 11:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)
        child14 = problem.add_object("child14", Child)
        child15 = problem.add_object("child15", Child)
        child16 = problem.add_object("child16", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)
        bread14 = problem.add_object("bread14", BreadPortion)
        bread15 = problem.add_object("bread15", BreadPortion)
        bread16 = problem.add_object("bread16", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)
        content14 = problem.add_object("content14", ContentPortion)
        content15 = problem.add_object("content15", ContentPortion)
        content16 = problem.add_object("content16", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)
        sandw18 = problem.add_object("sandw18", Sandwich)
        sandw19 = problem.add_object("sandw19", Sandwich)
        sandw20 = problem.add_object("sandw20", Sandwich)
        sandw21 = problem.add_object("sandw21", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)
        problem.set_initial_value(at_kitchen_bread(bread14), True)
        problem.set_initial_value(at_kitchen_bread(bread15), True)
        problem.set_initial_value(at_kitchen_bread(bread16), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)
        problem.set_initial_value(at_kitchen_content(content14), True)
        problem.set_initial_value(at_kitchen_content(content15), True)
        problem.set_initial_value(at_kitchen_content(content16), True)

        problem.set_initial_value(no_gluten_bread(bread4), True)
        problem.set_initial_value(no_gluten_bread(bread15), True)
        problem.set_initial_value(no_gluten_bread(bread6), True)
        problem.set_initial_value(no_gluten_bread(bread7), True)
        problem.set_initial_value(no_gluten_bread(bread2), True)
        problem.set_initial_value(no_gluten_bread(bread9), True)
        problem.set_initial_value(no_gluten_content(content8), True)
        problem.set_initial_value(no_gluten_content(content3), True)
        problem.set_initial_value(no_gluten_content(content14), True)
        problem.set_initial_value(no_gluten_content(content6), True)
        problem.set_initial_value(no_gluten_content(content1), True)
        problem.set_initial_value(no_gluten_content(content4), True)

        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(allergic_gluten(child10), True)
        problem.set_initial_value(allergic_gluten(child11), True)
        problem.set_initial_value(allergic_gluten(child5), True)
        problem.set_initial_value(allergic_gluten(child7), True)
        problem.set_initial_value(allergic_gluten(child9), True)
        problem.set_initial_value(not_allergic_gluten(child12), True)
        problem.set_initial_value(not_allergic_gluten(child13), True)
        problem.set_initial_value(not_allergic_gluten(child2), True)
        problem.set_initial_value(not_allergic_gluten(child3), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)
        problem.set_initial_value(not_allergic_gluten(child14), True)
        problem.set_initial_value(not_allergic_gluten(child6), True)
        problem.set_initial_value(not_allergic_gluten(child15), True)
        problem.set_initial_value(not_allergic_gluten(child8), True)
        problem.set_initial_value(not_allergic_gluten(child16), True)

        problem.set_initial_value(waiting(child1, table3), True)
        problem.set_initial_value(waiting(child2, table2), True)
        problem.set_initial_value(waiting(child3, table1), True)
        problem.set_initial_value(waiting(child4, table3), True)
        problem.set_initial_value(waiting(child5, table1), True)
        problem.set_initial_value(waiting(child6, table1), True)
        problem.set_initial_value(waiting(child7, table3), True)
        problem.set_initial_value(waiting(child8, table1), True)
        problem.set_initial_value(waiting(child9, table1), True)
        problem.set_initial_value(waiting(child10, table1), True)
        problem.set_initial_value(waiting(child11, table2), True)
        problem.set_initial_value(waiting(child12, table3), True)
        problem.set_initial_value(waiting(child13, table1), True)
        problem.set_initial_value(waiting(child14, table2), True)
        problem.set_initial_value(waiting(child15, table2), True)
        problem.set_initial_value(waiting(child16, table3), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)
        problem.set_initial_value(notexist(sandw18), True)
        problem.set_initial_value(notexist(sandw19), True)
        problem.set_initial_value(notexist(sandw20), True)
        problem.set_initial_value(notexist(sandw21), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
            served(child14),
            served(child15),
            served(child16),
        )
        problem.set_goal_network(gn)

    if problem_instance == 12:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)
        child14 = problem.add_object("child14", Child)
        child15 = problem.add_object("child15", Child)
        child16 = problem.add_object("child16", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)
        bread14 = problem.add_object("bread14", BreadPortion)
        bread15 = problem.add_object("bread15", BreadPortion)
        bread16 = problem.add_object("bread16", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)
        content14 = problem.add_object("content14", ContentPortion)
        content15 = problem.add_object("content15", ContentPortion)
        content16 = problem.add_object("content16", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)
        sandw18 = problem.add_object("sandw18", Sandwich)
        sandw19 = problem.add_object("sandw19", Sandwich)
        sandw20 = problem.add_object("sandw20", Sandwich)
        sandw21 = problem.add_object("sandw21", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)
        problem.set_initial_value(at_kitchen_bread(bread14), True)
        problem.set_initial_value(at_kitchen_bread(bread15), True)
        problem.set_initial_value(at_kitchen_bread(bread16), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)
        problem.set_initial_value(at_kitchen_content(content14), True)
        problem.set_initial_value(at_kitchen_content(content15), True)
        problem.set_initial_value(at_kitchen_content(content16), True)

        problem.set_initial_value(no_gluten_bread(bread10), True)
        problem.set_initial_value(no_gluten_bread(bread3), True)
        problem.set_initial_value(no_gluten_bread(bread14), True)
        problem.set_initial_value(no_gluten_bread(bread13), True)
        problem.set_initial_value(no_gluten_bread(bread6), True)
        problem.set_initial_value(no_gluten_bread(bread8), True)
        problem.set_initial_value(no_gluten_content(content4), True)
        problem.set_initial_value(no_gluten_content(content16), True)
        problem.set_initial_value(no_gluten_content(content5), True)
        problem.set_initial_value(no_gluten_content(content10), True)
        problem.set_initial_value(no_gluten_content(content3), True)
        problem.set_initial_value(no_gluten_content(content11), True)

        problem.set_initial_value(allergic_gluten(child2), True)
        problem.set_initial_value(allergic_gluten(child3), True)
        problem.set_initial_value(allergic_gluten(child16), True)
        problem.set_initial_value(allergic_gluten(child5), True)
        problem.set_initial_value(allergic_gluten(child6), True)
        problem.set_initial_value(allergic_gluten(child11), True)
        problem.set_initial_value(not_allergic_gluten(child12), True)
        problem.set_initial_value(not_allergic_gluten(child1), True)
        problem.set_initial_value(not_allergic_gluten(child10), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)
        problem.set_initial_value(not_allergic_gluten(child14), True)
        problem.set_initial_value(not_allergic_gluten(child7), True)
        problem.set_initial_value(not_allergic_gluten(child8), True)
        problem.set_initial_value(not_allergic_gluten(child9), True)
        problem.set_initial_value(not_allergic_gluten(child15), True)
        problem.set_initial_value(not_allergic_gluten(child13), True)

        problem.set_initial_value(waiting(child1, table1), True)
        problem.set_initial_value(waiting(child2, table2), True)
        problem.set_initial_value(waiting(child3, table2), True)
        problem.set_initial_value(waiting(child4, table1), True)
        problem.set_initial_value(waiting(child5, table2), True)
        problem.set_initial_value(waiting(child6, table1), True)
        problem.set_initial_value(waiting(child7, table1), True)
        problem.set_initial_value(waiting(child8, table3), True)
        problem.set_initial_value(waiting(child9, table3), True)
        problem.set_initial_value(waiting(child10, table2), True)
        problem.set_initial_value(waiting(child11, table3), True)
        problem.set_initial_value(waiting(child12, table2), True)
        problem.set_initial_value(waiting(child13, table2), True)
        problem.set_initial_value(waiting(child14, table2), True)
        problem.set_initial_value(waiting(child15, table2), True)
        problem.set_initial_value(waiting(child16, table3), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)
        problem.set_initial_value(notexist(sandw18), True)
        problem.set_initial_value(notexist(sandw19), True)
        problem.set_initial_value(notexist(sandw20), True)
        problem.set_initial_value(notexist(sandw21), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
            served(child14),
            served(child15),
            served(child16),
        )
        problem.set_goal_network(gn)

    if problem_instance == 13:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)
        child14 = problem.add_object("child14", Child)
        child15 = problem.add_object("child15", Child)
        child16 = problem.add_object("child16", Child)
        child17 = problem.add_object("child17", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)
        bread14 = problem.add_object("bread14", BreadPortion)
        bread15 = problem.add_object("bread15", BreadPortion)
        bread16 = problem.add_object("bread16", BreadPortion)
        bread17 = problem.add_object("bread17", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)
        content14 = problem.add_object("content14", ContentPortion)
        content15 = problem.add_object("content15", ContentPortion)
        content16 = problem.add_object("content16", ContentPortion)
        content17 = problem.add_object("content17", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)
        tray4 = problem.add_object("tray4", Tray)  # New tray object

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)
        sandw18 = problem.add_object("sandw18", Sandwich)
        sandw19 = problem.add_object("sandw19", Sandwich)
        sandw20 = problem.add_object("sandw20", Sandwich)
        sandw21 = problem.add_object("sandw21", Sandwich)
        sandw22 = problem.add_object("sandw22", Sandwich)
        sandw23 = problem.add_object("sandw23", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)
        problem.set_initial_value(at(tray4, kitchen), True)  # New tray initial state

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)
        problem.set_initial_value(at_kitchen_bread(bread14), True)
        problem.set_initial_value(at_kitchen_bread(bread15), True)
        problem.set_initial_value(at_kitchen_bread(bread16), True)
        problem.set_initial_value(at_kitchen_bread(bread17), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)
        problem.set_initial_value(at_kitchen_content(content14), True)
        problem.set_initial_value(at_kitchen_content(content15), True)
        problem.set_initial_value(at_kitchen_content(content16), True)
        problem.set_initial_value(at_kitchen_content(content17), True)

        problem.set_initial_value(no_gluten_bread(bread10), True)
        problem.set_initial_value(no_gluten_bread(bread4), True)
        problem.set_initial_value(no_gluten_bread(bread15), True)
        problem.set_initial_value(no_gluten_bread(bread13), True)
        problem.set_initial_value(no_gluten_bread(bread7), True)
        problem.set_initial_value(no_gluten_bread(bread8), True)
        problem.set_initial_value(no_gluten_content(content4), True)
        problem.set_initial_value(no_gluten_content(content17), True)
        problem.set_initial_value(no_gluten_content(content5), True)
        problem.set_initial_value(no_gluten_content(content10), True)
        problem.set_initial_value(no_gluten_content(content3), True)
        problem.set_initial_value(no_gluten_content(content12), True)

        problem.set_initial_value(allergic_gluten(child2), True)
        problem.set_initial_value(allergic_gluten(child11), True)
        problem.set_initial_value(allergic_gluten(child4), True)
        problem.set_initial_value(allergic_gluten(child17), True)
        problem.set_initial_value(allergic_gluten(child6), True)
        problem.set_initial_value(allergic_gluten(child5), True)
        problem.set_initial_value(not_allergic_gluten(child12), True)
        problem.set_initial_value(not_allergic_gluten(child1), True)
        problem.set_initial_value(not_allergic_gluten(child10), True)
        problem.set_initial_value(not_allergic_gluten(child3), True)
        problem.set_initial_value(not_allergic_gluten(child16), True)
        problem.set_initial_value(not_allergic_gluten(child14), True)
        problem.set_initial_value(not_allergic_gluten(child7), True)
        problem.set_initial_value(not_allergic_gluten(child8), True)
        problem.set_initial_value(not_allergic_gluten(child9), True)
        problem.set_initial_value(not_allergic_gluten(child15), True)
        problem.set_initial_value(not_allergic_gluten(child13), True)

        problem.set_initial_value(waiting(child1, table1), True)
        problem.set_initial_value(waiting(child2, table2), True)
        problem.set_initial_value(waiting(child3, table2), True)
        problem.set_initial_value(waiting(child4, table1), True)
        problem.set_initial_value(waiting(child5, table2), True)
        problem.set_initial_value(waiting(child6, table1), True)
        problem.set_initial_value(waiting(child7, table1), True)
        problem.set_initial_value(waiting(child8, table3), True)
        problem.set_initial_value(waiting(child9, table3), True)
        problem.set_initial_value(waiting(child10, table2), True)
        problem.set_initial_value(waiting(child11, table3), True)
        problem.set_initial_value(waiting(child12, table2), True)
        problem.set_initial_value(waiting(child13, table2), True)
        problem.set_initial_value(waiting(child14, table2), True)
        problem.set_initial_value(waiting(child15, table2), True)
        problem.set_initial_value(waiting(child16, table3), True)
        problem.set_initial_value(waiting(child17, table1), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)
        problem.set_initial_value(notexist(sandw18), True)
        problem.set_initial_value(notexist(sandw19), True)
        problem.set_initial_value(notexist(sandw20), True)
        problem.set_initial_value(notexist(sandw21), True)
        problem.set_initial_value(notexist(sandw22), True)
        problem.set_initial_value(notexist(sandw23), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
            served(child14),
            served(child15),
            served(child16),
            served(child17),
        )
        problem.set_goal_network(gn)

    if problem_instance == 14:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)
        child14 = problem.add_object("child14", Child)
        child15 = problem.add_object("child15", Child)
        child16 = problem.add_object("child16", Child)
        child17 = problem.add_object("child17", Child)
        child18 = problem.add_object("child18", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)
        bread14 = problem.add_object("bread14", BreadPortion)
        bread15 = problem.add_object("bread15", BreadPortion)
        bread16 = problem.add_object("bread16", BreadPortion)
        bread17 = problem.add_object("bread17", BreadPortion)
        bread18 = problem.add_object("bread18", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)
        content14 = problem.add_object("content14", ContentPortion)
        content15 = problem.add_object("content15", ContentPortion)
        content16 = problem.add_object("content16", ContentPortion)
        content17 = problem.add_object("content17", ContentPortion)
        content18 = problem.add_object("content18", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)
        tray4 = problem.add_object("tray4", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)
        sandw18 = problem.add_object("sandw18", Sandwich)
        sandw19 = problem.add_object("sandw19", Sandwich)
        sandw20 = problem.add_object("sandw20", Sandwich)
        sandw21 = problem.add_object("sandw21", Sandwich)
        sandw22 = problem.add_object("sandw22", Sandwich)
        sandw23 = problem.add_object("sandw23", Sandwich)
        sandw24 = problem.add_object("sandw24", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)
        problem.set_initial_value(at(tray4, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)
        problem.set_initial_value(at_kitchen_bread(bread14), True)
        problem.set_initial_value(at_kitchen_bread(bread15), True)
        problem.set_initial_value(at_kitchen_bread(bread16), True)
        problem.set_initial_value(at_kitchen_bread(bread17), True)
        problem.set_initial_value(at_kitchen_bread(bread18), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)
        problem.set_initial_value(at_kitchen_content(content14), True)
        problem.set_initial_value(at_kitchen_content(content15), True)
        problem.set_initial_value(at_kitchen_content(content16), True)
        problem.set_initial_value(at_kitchen_content(content17), True)
        problem.set_initial_value(at_kitchen_content(content18), True)

        problem.set_initial_value(no_gluten_bread(bread4), True)
        problem.set_initial_value(no_gluten_bread(bread17), True)
        problem.set_initial_value(no_gluten_bread(bread7), True)
        problem.set_initial_value(no_gluten_bread(bread8), True)
        problem.set_initial_value(no_gluten_bread(bread2), True)
        problem.set_initial_value(no_gluten_bread(bread11), True)
        problem.set_initial_value(no_gluten_bread(bread6), True)
        problem.set_initial_value(no_gluten_content(content3), True)
        problem.set_initial_value(no_gluten_content(content17), True)
        problem.set_initial_value(no_gluten_content(content7), True)
        problem.set_initial_value(no_gluten_content(content1), True)
        problem.set_initial_value(no_gluten_content(content5), True)
        problem.set_initial_value(no_gluten_content(content9), True)
        problem.set_initial_value(no_gluten_content(content4), True)

        problem.set_initial_value(allergic_gluten(child12), True)
        problem.set_initial_value(allergic_gluten(child1), True)
        problem.set_initial_value(allergic_gluten(child14), True)
        problem.set_initial_value(allergic_gluten(child13), True)
        problem.set_initial_value(allergic_gluten(child8), True)
        problem.set_initial_value(allergic_gluten(child9), True)
        problem.set_initial_value(allergic_gluten(child18), True)
        problem.set_initial_value(not_allergic_gluten(child2), True)
        problem.set_initial_value(not_allergic_gluten(child3), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)
        problem.set_initial_value(not_allergic_gluten(child5), True)
        problem.set_initial_value(not_allergic_gluten(child6), True)
        problem.set_initial_value(not_allergic_gluten(child7), True)
        problem.set_initial_value(not_allergic_gluten(child15), True)
        problem.set_initial_value(not_allergic_gluten(child10), True)
        problem.set_initial_value(not_allergic_gluten(child11), True)
        problem.set_initial_value(not_allergic_gluten(child16), True)
        problem.set_initial_value(not_allergic_gluten(child17), True)

        problem.set_initial_value(waiting(child1, table3), True)
        problem.set_initial_value(waiting(child2, table1), True)
        problem.set_initial_value(waiting(child3, table1), True)
        problem.set_initial_value(waiting(child4, table3), True)
        problem.set_initial_value(waiting(child5, table1), True)
        problem.set_initial_value(waiting(child6, table1), True)
        problem.set_initial_value(waiting(child7, table1), True)
        problem.set_initial_value(waiting(child8, table2), True)
        problem.set_initial_value(waiting(child9, table3), True)
        problem.set_initial_value(waiting(child10, table1), True)
        problem.set_initial_value(waiting(child11, table2), True)
        problem.set_initial_value(waiting(child12, table2), True)
        problem.set_initial_value(waiting(child13, table3), True)
        problem.set_initial_value(waiting(child14, table3), True)
        problem.set_initial_value(waiting(child15, table1), True)
        problem.set_initial_value(waiting(child16, table2), True)
        problem.set_initial_value(waiting(child17, table2), True)
        problem.set_initial_value(waiting(child18, table3), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)
        problem.set_initial_value(notexist(sandw18), True)
        problem.set_initial_value(notexist(sandw19), True)
        problem.set_initial_value(notexist(sandw20), True)
        problem.set_initial_value(notexist(sandw21), True)
        problem.set_initial_value(notexist(sandw22), True)
        problem.set_initial_value(notexist(sandw23), True)
        problem.set_initial_value(notexist(sandw24), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
            served(child14),
            served(child15),
            served(child16),
            served(child17),
            served(child18),
        )
        problem.set_goal_network(gn)

    if problem_instance == 15:
        # 1. Define Objects
        child1 = problem.add_object("child1", Child)
        child2 = problem.add_object("child2", Child)
        child3 = problem.add_object("child3", Child)
        child4 = problem.add_object("child4", Child)
        child5 = problem.add_object("child5", Child)
        child6 = problem.add_object("child6", Child)
        child7 = problem.add_object("child7", Child)
        child8 = problem.add_object("child8", Child)
        child9 = problem.add_object("child9", Child)
        child10 = problem.add_object("child10", Child)
        child11 = problem.add_object("child11", Child)
        child12 = problem.add_object("child12", Child)
        child13 = problem.add_object("child13", Child)
        child14 = problem.add_object("child14", Child)
        child15 = problem.add_object("child15", Child)
        child16 = problem.add_object("child16", Child)
        child17 = problem.add_object("child17", Child)
        child18 = problem.add_object("child18", Child)

        bread1 = problem.add_object("bread1", BreadPortion)
        bread2 = problem.add_object("bread2", BreadPortion)
        bread3 = problem.add_object("bread3", BreadPortion)
        bread4 = problem.add_object("bread4", BreadPortion)
        bread5 = problem.add_object("bread5", BreadPortion)
        bread6 = problem.add_object("bread6", BreadPortion)
        bread7 = problem.add_object("bread7", BreadPortion)
        bread8 = problem.add_object("bread8", BreadPortion)
        bread9 = problem.add_object("bread9", BreadPortion)
        bread10 = problem.add_object("bread10", BreadPortion)
        bread11 = problem.add_object("bread11", BreadPortion)
        bread12 = problem.add_object("bread12", BreadPortion)
        bread13 = problem.add_object("bread13", BreadPortion)
        bread14 = problem.add_object("bread14", BreadPortion)
        bread15 = problem.add_object("bread15", BreadPortion)
        bread16 = problem.add_object("bread16", BreadPortion)
        bread17 = problem.add_object("bread17", BreadPortion)
        bread18 = problem.add_object("bread18", BreadPortion)

        content1 = problem.add_object("content1", ContentPortion)
        content2 = problem.add_object("content2", ContentPortion)
        content3 = problem.add_object("content3", ContentPortion)
        content4 = problem.add_object("content4", ContentPortion)
        content5 = problem.add_object("content5", ContentPortion)
        content6 = problem.add_object("content6", ContentPortion)
        content7 = problem.add_object("content7", ContentPortion)
        content8 = problem.add_object("content8", ContentPortion)
        content9 = problem.add_object("content9", ContentPortion)
        content10 = problem.add_object("content10", ContentPortion)
        content11 = problem.add_object("content11", ContentPortion)
        content12 = problem.add_object("content12", ContentPortion)
        content13 = problem.add_object("content13", ContentPortion)
        content14 = problem.add_object("content14", ContentPortion)
        content15 = problem.add_object("content15", ContentPortion)
        content16 = problem.add_object("content16", ContentPortion)
        content17 = problem.add_object("content17", ContentPortion)
        content18 = problem.add_object("content18", ContentPortion)

        tray1 = problem.add_object("tray1", Tray)
        tray2 = problem.add_object("tray2", Tray)
        tray3 = problem.add_object("tray3", Tray)
        tray4 = problem.add_object("tray4", Tray)

        table1 = problem.add_object("table1", Place)
        table2 = problem.add_object("table2", Place)
        table3 = problem.add_object("table3", Place)

        sandw1 = problem.add_object("sandw1", Sandwich)
        sandw2 = problem.add_object("sandw2", Sandwich)
        sandw3 = problem.add_object("sandw3", Sandwich)
        sandw4 = problem.add_object("sandw4", Sandwich)
        sandw5 = problem.add_object("sandw5", Sandwich)
        sandw6 = problem.add_object("sandw6", Sandwich)
        sandw7 = problem.add_object("sandw7", Sandwich)
        sandw8 = problem.add_object("sandw8", Sandwich)
        sandw9 = problem.add_object("sandw9", Sandwich)
        sandw10 = problem.add_object("sandw10", Sandwich)
        sandw11 = problem.add_object("sandw11", Sandwich)
        sandw12 = problem.add_object("sandw12", Sandwich)
        sandw13 = problem.add_object("sandw13", Sandwich)
        sandw14 = problem.add_object("sandw14", Sandwich)
        sandw15 = problem.add_object("sandw15", Sandwich)
        sandw16 = problem.add_object("sandw16", Sandwich)
        sandw17 = problem.add_object("sandw17", Sandwich)
        sandw18 = problem.add_object("sandw18", Sandwich)
        sandw19 = problem.add_object("sandw19", Sandwich)
        sandw20 = problem.add_object("sandw20", Sandwich)
        sandw21 = problem.add_object("sandw21", Sandwich)
        sandw22 = problem.add_object("sandw22", Sandwich)
        sandw23 = problem.add_object("sandw23", Sandwich)
        sandw24 = problem.add_object("sandw24", Sandwich)

        # 2. Set Initial State
        problem.set_initial_value(at(tray1, kitchen), True)
        problem.set_initial_value(at(tray2, kitchen), True)
        problem.set_initial_value(at(tray3, kitchen), True)
        problem.set_initial_value(at(tray4, kitchen), True)

        problem.set_initial_value(at_kitchen_bread(bread1), True)
        problem.set_initial_value(at_kitchen_bread(bread2), True)
        problem.set_initial_value(at_kitchen_bread(bread3), True)
        problem.set_initial_value(at_kitchen_bread(bread4), True)
        problem.set_initial_value(at_kitchen_bread(bread5), True)
        problem.set_initial_value(at_kitchen_bread(bread6), True)
        problem.set_initial_value(at_kitchen_bread(bread7), True)
        problem.set_initial_value(at_kitchen_bread(bread8), True)
        problem.set_initial_value(at_kitchen_bread(bread9), True)
        problem.set_initial_value(at_kitchen_bread(bread10), True)
        problem.set_initial_value(at_kitchen_bread(bread11), True)
        problem.set_initial_value(at_kitchen_bread(bread12), True)
        problem.set_initial_value(at_kitchen_bread(bread13), True)
        problem.set_initial_value(at_kitchen_bread(bread14), True)
        problem.set_initial_value(at_kitchen_bread(bread15), True)
        problem.set_initial_value(at_kitchen_bread(bread16), True)
        problem.set_initial_value(at_kitchen_bread(bread17), True)
        problem.set_initial_value(at_kitchen_bread(bread18), True)

        problem.set_initial_value(at_kitchen_content(content1), True)
        problem.set_initial_value(at_kitchen_content(content2), True)
        problem.set_initial_value(at_kitchen_content(content3), True)
        problem.set_initial_value(at_kitchen_content(content4), True)
        problem.set_initial_value(at_kitchen_content(content5), True)
        problem.set_initial_value(at_kitchen_content(content6), True)
        problem.set_initial_value(at_kitchen_content(content7), True)
        problem.set_initial_value(at_kitchen_content(content8), True)
        problem.set_initial_value(at_kitchen_content(content9), True)
        problem.set_initial_value(at_kitchen_content(content10), True)
        problem.set_initial_value(at_kitchen_content(content11), True)
        problem.set_initial_value(at_kitchen_content(content12), True)
        problem.set_initial_value(at_kitchen_content(content13), True)
        problem.set_initial_value(at_kitchen_content(content14), True)
        problem.set_initial_value(at_kitchen_content(content15), True)
        problem.set_initial_value(at_kitchen_content(content16), True)
        problem.set_initial_value(at_kitchen_content(content17), True)
        problem.set_initial_value(at_kitchen_content(content18), True)

        problem.set_initial_value(no_gluten_bread(bread11), True)
        problem.set_initial_value(no_gluten_bread(bread4), True)
        problem.set_initial_value(no_gluten_bread(bread16), True)
        problem.set_initial_value(no_gluten_bread(bread14), True)
        problem.set_initial_value(no_gluten_bread(bread7), True)
        problem.set_initial_value(no_gluten_bread(bread9), True)
        problem.set_initial_value(no_gluten_bread(bread3), True)
        problem.set_initial_value(no_gluten_content(content4), True)
        problem.set_initial_value(no_gluten_content(content5), True)
        problem.set_initial_value(no_gluten_content(content12), True)
        problem.set_initial_value(no_gluten_content(content18), True)
        problem.set_initial_value(no_gluten_content(content14), True)
        problem.set_initial_value(no_gluten_content(content1), True)
        problem.set_initial_value(no_gluten_content(content3), True)

        problem.set_initial_value(allergic_gluten(child17), True)
        problem.set_initial_value(allergic_gluten(child2), True)
        problem.set_initial_value(allergic_gluten(child14), True)
        problem.set_initial_value(allergic_gluten(child6), True)
        problem.set_initial_value(allergic_gluten(child8), True)
        problem.set_initial_value(allergic_gluten(child9), True)
        problem.set_initial_value(allergic_gluten(child5), True)
        problem.set_initial_value(not_allergic_gluten(child12), True)
        problem.set_initial_value(not_allergic_gluten(child1), True)
        problem.set_initial_value(not_allergic_gluten(child10), True)
        problem.set_initial_value(not_allergic_gluten(child3), True)
        problem.set_initial_value(not_allergic_gluten(child4), True)
        problem.set_initial_value(not_allergic_gluten(child7), True)
        problem.set_initial_value(not_allergic_gluten(child13), True)
        problem.set_initial_value(not_allergic_gluten(child18), True)
        problem.set_initial_value(not_allergic_gluten(child15), True)
        problem.set_initial_value(not_allergic_gluten(child11), True)
        problem.set_initial_value(not_allergic_gluten(child16), True)

        problem.set_initial_value(waiting(child1, table1), True)
        problem.set_initial_value(waiting(child2, table2), True)
        problem.set_initial_value(waiting(child3, table1), True)
        problem.set_initial_value(waiting(child4, table1), True)
        problem.set_initial_value(waiting(child5, table3), True)
        problem.set_initial_value(waiting(child6, table3), True)
        problem.set_initial_value(waiting(child7, table2), True)
        problem.set_initial_value(waiting(child8, table3), True)
        problem.set_initial_value(waiting(child9, table2), True)
        problem.set_initial_value(waiting(child10, table2), True)
        problem.set_initial_value(waiting(child11, table2), True)
        problem.set_initial_value(waiting(child12, table2), True)
        problem.set_initial_value(waiting(child13, table3), True)
        problem.set_initial_value(waiting(child14, table1), True)
        problem.set_initial_value(waiting(child15, table2), True)
        problem.set_initial_value(waiting(child16, table2), True)
        problem.set_initial_value(waiting(child17, table2), True)
        problem.set_initial_value(waiting(child18, table1), True)

        problem.set_initial_value(notexist(sandw1), True)
        problem.set_initial_value(notexist(sandw2), True)
        problem.set_initial_value(notexist(sandw3), True)
        problem.set_initial_value(notexist(sandw4), True)
        problem.set_initial_value(notexist(sandw5), True)
        problem.set_initial_value(notexist(sandw6), True)
        problem.set_initial_value(notexist(sandw7), True)
        problem.set_initial_value(notexist(sandw8), True)
        problem.set_initial_value(notexist(sandw9), True)
        problem.set_initial_value(notexist(sandw10), True)
        problem.set_initial_value(notexist(sandw11), True)
        problem.set_initial_value(notexist(sandw12), True)
        problem.set_initial_value(notexist(sandw13), True)
        problem.set_initial_value(notexist(sandw14), True)
        problem.set_initial_value(notexist(sandw15), True)
        problem.set_initial_value(notexist(sandw16), True)
        problem.set_initial_value(notexist(sandw17), True)
        problem.set_initial_value(notexist(sandw18), True)
        problem.set_initial_value(notexist(sandw19), True)
        problem.set_initial_value(notexist(sandw20), True)
        problem.set_initial_value(notexist(sandw21), True)
        problem.set_initial_value(notexist(sandw22), True)
        problem.set_initial_value(notexist(sandw23), True)
        problem.set_initial_value(notexist(sandw24), True)

        # 3. Set Goal Network
        gn = PartialOrderGoalNetwork()
        gn.add(
            served(child1),
            served(child2),
            served(child3),
            served(child4),
            served(child5),
            served(child6),
            served(child7),
            served(child8),
            served(child9),
            served(child10),
            served(child11),
            served(child12),
            served(child13),
            served(child14),
            served(child15),
            served(child16),
            served(child17),
            served(child18),
        )
        problem.set_goal_network(gn)

    return problem
