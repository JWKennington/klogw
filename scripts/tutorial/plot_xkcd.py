"""Tutorial for using XKCD style on a matplotlib plot
"""

import matplotlib.pyplot as plt


def main():
    """Main function"""
    # Set the style to XKCD
    plt.xkcd()

    # Create a simple plot
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])

    # Add a title
    plt.title('Simple plot')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()
