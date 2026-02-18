import numpy as np
import random


class Card:
    def __init__(self):
        # все числа на карточке уникальны
        self.matrix = np.sort(np.random.choice(90, 15, replace=False)) \
                      .reshape(3, 5)

    def checker(self, num):
        """Функция для проверки наличия числа в карточке"""
        return num in self.matrix

    def print_card(self):
        """Функция для печати карточки"""
        for row in self.matrix:
            print(*row)
        print("--------------------------")

    def cross_out(self, value):
        """Функция для вычеркивания числа из карточки"""
        # заменяем число на None, т.к. если заменять на строку, то изменится
        # тип данных массива, что приведет к усложнению проверки на наличие
        # числа
        self.matrix = np.where(self.matrix == value, None, self.matrix)

    def check_win(self):
        """Функция для проверки того, что все числа зачеркнуты"""
        return not self.matrix.any() 


class Barrel:
    counter = 0
    numbers_in_game = list(range(1, 91))

    def __init__(self):
        Barrel.counter += 1
        self.number = random.choice(Barrel.numbers_in_game)
        Barrel.numbers_in_game.remove(self.number)


class Game:
    def start_game(self):
        """Функция для запуска игры"""
        print("Игра началась!")
        self.card_player = Card()
        self.card_pc = Card()

        # цикл игры
        for i in range(1, 91):
            barrel = Barrel()
            print(f"Новый бочонок: {barrel.number} (осталось {90 - i}) ")

            print("------ Ваша карточка -----")
            self.card_player.print_card()
            print("-- Карточка компьютера ---")
            self.card_pc.print_card()

            # ход игрока
            while (answer := input("Зачеркнуть число? (y/n)")) not in ("y", "n"):
                print("Введите корректный ответ")

            # проверяем, есть ли число в карточке игрока
            check_player = self.card_player.checker(barrel.number)
            if answer == "y":
                if check_player:
                    # вычеркиваем число
                    self.card_player.cross_out(barrel.number)
                else:
                    print("Вы проиграли")
                    break

            else:
                if check_player:
                    print("Вы проиграли")
                    break

            # ход компьютера
            # проверяем, есть ли число в карточке компьютера
            check_pc = self.card_pc.checker(barrel.number)
            if check_pc:
                # вычеркиваем число
                self.card_pc.cross_out(barrel.number)

            # проверка, что все числа игрока зачеркнуты
            if self.card_player.check_win():
                print('Вы победили!')
                break
            # проверка, что все числа компьютера зачеркнуты
            if self.card_pc.check_win():
                print('Вы проиграли!')
                break


game = Game()
game.start_game()