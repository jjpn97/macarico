from macarico.tasks.mountain_car import MountainCar
from macarico.tasks.mountain_car import MountainCarLoss
from macarico.tasks.mountain_car import MountainCarFeatures
from macarico.features.actor import TransitionBOW
from macarico.features.sequence import AttendAt
from test_pocman import run_environment


def test():
    print('')
    print('Mountain Car')
    print('')
    ex = MountainCar(T=100)
    run_environment(
        ex,
        lambda:
        TransitionBOW([MountainCarFeatures()],
                      [AttendAt(lambda _: 0, 'mountain_car')],
                      ex.n_actions),
        MountainCarLoss(),
        n_epochs=201,
    )

if __name__ == '__main__':
    test()
