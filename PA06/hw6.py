import re


def problem1(searchstring):
    """
    Match emails.

    :param searchstring: string
    :return: string
    """
    pass


def problem2(searchstring):
    """
    Extract author and book.

    :param searchstring: string
    :return: tuple
    """
    pass


def problem3(searchstring):
    """
    Replace Boy/Girl or boy/girl with Man/Woman or man/woman respectively.

    :param searchstring: string
    :return: string
    """
    pass


if __name__ == '__main__':

    print("\nProblem 1:")
    testcase11 = "starlight.123@vought.com"
    print(
        "Student answer: ",
        problem1(testcase11),
        "\tAnswer correct?",
        problem1(testcase11) == "valid",
    )

    testcase12 = "hughie.250supe@fbsa.gov"
    print(
        "Student answer: ",
        problem1(testcase12),
        "\tAnswer correct?",
        problem1(testcase12) == "valid",
    )

    testcase13 = "marie.100@godolkin.edu"
    print(
        "Student answer: ",
        problem1(testcase13),
        "\tAnswer correct?",
        problem1(testcase13) == "valid",
    )

    testcase14 = "a-train.123@vought.com"
    print(
        "Student answer: ",
        problem1(testcase14),
        "\tAnswer correct?",
        problem1(testcase14) == "invalid",# hyphen in name
    )

    testcase15 = "homelander.800@vought.com"
    print(
        "Student answer: ",
        problem1(testcase15),
        "\tAnswer correct?",
        problem1(testcase15) == "invalid",# out of range
    )

    testcase16 = "blacknoir.567*abc@godolkin.edu"
    print(
        "Student answer: ",
        problem1(testcase16),
        "\tAnswer correct?",
        problem1(testcase16) == "invalid",  # non-letters after id
    )

    testcase17 = "noir.123@vought.co"
    print(
        "Student answer: ",
        problem1(testcase17),
        "\tAnswer correct?",
        problem1(testcase17) == "invalid",
    )

    testcase18 = "starlight.144@vought.comasdf"
    print(
        "Student answer: ",
        problem1(testcase18),
        "\tAnswer correct?",
        problem1(testcase18) == "invalid",
    )


    print("\nProblem 2:")
    testcase21 = "George Orwell wrote 1984"
    print(
        "Student answer: ",
        problem2(testcase21),
        "\tAnswer correct?",
        problem2(testcase21) == ("George Orwell", "1984"),
    )

    testcase22 = "In the 1930s, a Mystery writer wrote Mary Westmacotts. Later it was found that Agatha Christie wrote The Westmacott Novels"
    print(
        "Student answer: ",
        problem2(testcase22),
        "\tAnswer correct?",
        problem2(testcase22) == ("Agatha Christie", "The Westmacott Novels"),
    )

    testcase23 = "Roxette wrote books"
    print(
        "Student answer: ",
        problem2(testcase23),
        "\tAnswer correct?",
        problem2(testcase23) == ("Roxette", "books"),
    )

    testcase24 = "Erin Morgenstern wrote The Starless Sea Book and The Night Circus"
    print(
        "Student answer: ",
        problem2(testcase24),
        "\tAnswer correct?",
        problem2(testcase24) == ("Erin Morgenstern", "The Starless Sea"),
    )

    testcase25 = "Haruki Murakami wrote 1Q84"
    print(
        "Student answer: ",
        problem2(testcase25),
        "\tAnswer correct?",
        problem2(testcase25) == ("Haruki Murakami", "1Q84"),
    )

    testcase26 = "Khaled Hosseini wrote sad books"
    print(
        "Student answer: ",
        problem2(testcase26),
        "\tAnswer correct?",
        problem2(testcase26) == ("noauthor", "noname"),
    )

    testcase27 = "Haruki Murakami wrote Norwegian Wood"
    print(
        "Student answer: ",
        problem2(testcase27),
        "\tAnswer correct?",
        problem2(testcase27) == ("Haruki Murakami", "Norwegian Wood"),
    )

    print("\nProblem 3:")
    testcase31 = "Crimson Girl, report to Brink Hall!"
    print(
        "Student answer: ",
        problem3(testcase31),
        "\tAnswer correct?",
        problem3(testcase31) == "Crimson Woman, report to Brink Hall!",
    )

    testcase32 = "There is a girl in Dorm C Tech Boy"
    print(
        "Student answer: ",
        problem3(testcase32),
        "\tAnswer correct?",
        problem3(testcase32) == "There is a girl in Dorm C Tech Man",
    )

    testcase33 = "There is a boy that needs help"
    print(
        "Student answer: ",
        problem3(testcase33),
        "\tAnswer correct?",
        problem3(testcase33) == "nomatch",
    )
