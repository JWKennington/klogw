"""Post: Flavors of Physicists
"""

import matplotlib.pyplot as plt
import numpy
import scipy


def plot_theorist_types():
    """The below function makes a plot of the different types of physicists. Specific features of the plot include
    the following:

    - Format: XKCD style
    - Title: Types of Theorists in Physics
    - The origin should be centered on the plot
    - Y-axis label text top side: "Physics\n(Intuitive)"
    - Y-axis label text bottom side: "Mathematics\n(Formal)"
    - X-axis label text right: "Constructive"
    - X-axis label text left: "Analytical"
    - The plot should have the following points:
        - (1, 1): "Phenomenal\nPhysicist"
        - (1, -1): "Philosopher\nPhysicist"
        - (-1, -1): "Mathematical\nAnalyst"
        - (-1, 1): "Mathematical\nConstructor"
    """
    size_quadrant_name = 12
    size_quadrant_number = 16
    size_quadrant_example = 10
    scale_quadrant_number = 1.0
    offset_quadrant_name_y = 0.4
    offset_quadrant_example_y = -0.3

    # Set the style to XKCD
    plt.xkcd()

    # Hide plot borders
    plt.axis('off')

    # Show x and y axes
    plt.axhline(0, color='black', lw=2)
    plt.axvline(0, color='black', lw=2)

    # Center the plot
    plt.xlim(-1.75, 1.75)
    plt.ylim(-2, 2)

    # Add x-axis right text
    plt.text(2.7, 0, 'Constructive', ha='center', va='center')

    # Add x-axis left text
    plt.text(-2.5, 0, 'Analytical', ha='center', va='center')

    # Add y-axis top text
    plt.text(0, 2.1, 'Physics (Intuitive)', ha='center', va='center')

    # Add y-axis bottom text
    plt.text(0, -2.1, 'Mathematics (Formal)', ha='center', va='center')

    # Add the quadrant numbers
    plt.text(scale_quadrant_number * 1, scale_quadrant_number * 1 + offset_quadrant_name_y, '1', ha='center',
             va='center', size=size_quadrant_number, color='purple')
    plt.text(scale_quadrant_number * -1, scale_quadrant_number * 1 + offset_quadrant_name_y, '2', ha='center',
             va='center', size=size_quadrant_number, color='purple')
    plt.text(scale_quadrant_number * -1, scale_quadrant_number * -1 + offset_quadrant_name_y, '3', ha='center',
             va='center', size=size_quadrant_number, color='purple')
    plt.text(scale_quadrant_number * 1, scale_quadrant_number * -1 + offset_quadrant_name_y, '4', ha='center',
             va='center', size=size_quadrant_number, color='purple')

    # Add the quadrant names
    plt.text(0, 2.5, '(Examples from Loop Quantum Gravity)', ha='center', va='center', size=8, color='purple')
    plt.text(1, 1, 'Phenomenal\nPhysicist', ha='center', va='center', size=size_quadrant_name, color='blue')
    plt.text(-1, 1, 'Philosopher\nPhysicist', ha='center', va='center', size=size_quadrant_name, color='blue')
    plt.text(-1, -1, 'Mathematical\nAnalyst', ha='center', va='center', size=size_quadrant_name, color='blue')
    plt.text(1, -1, 'Mathematical\nConstructor', ha='center', va='center', size=size_quadrant_name, color='blue')

    # Add the quadrant examples
    plt.text(1, 1 + offset_quadrant_example_y, '(Smolin)', ha='center', va='top', size=size_quadrant_example, color='purple')  # Feynman
    plt.text(-1, 1 + offset_quadrant_example_y, '(Rovelli)', ha='center', va='top',
             size=size_quadrant_example, color='purple')  # Einstein
    plt.text(-1, -1 + offset_quadrant_example_y, '(Ashtekar)', ha='center', va='top',
             size=size_quadrant_example, color='purple')  # Bethe
    plt.text(1, -1 + offset_quadrant_example_y, '(Thiemann)', ha='center', va='top',
             size=size_quadrant_example, color='purple')  # Penrose

    # Add a title
    plt.title('Types of Theorists in Physics', pad=45)

    # Add more white space around plot
    plt.tight_layout(pad=0.5)

    # Show the plot
    plt.savefig('types_of_theorists.png', dpi=300)
    plt.show()


def plot_broad_types():
    """The below function makes a plot of the broad types of physicists. Specific features of the plot include
    the following:


    """
    # Configs
    size_name = 12
    size_y_types = 10
    offset_y_types = -0.5

    # Set the style to XKCD
    plt.xkcd()

    # Hide plot borders
    plt.axis('off')

    # Show x and y axes
    plt.axhline(0, color='black', lw=2)
    plt.axvline(0, color='black', lw=2)

    # Center the plot
    plt.xlim(-3, 3)
    plt.ylim(-2, 2)

    # Add x-axis right text
    plt.text(2.6, 0.15, 'Theory', ha='center', va='center')

    # Add x-axis left text
    plt.text(-2.4, 0.15, 'Experiment', ha='center', va='center')

    # Add x-axis middle text
    plt.text(0, 0.15, 'Phenomenology', ha='center', va='center')

    # Add y-axis top text
    plt.text(0, 2.2, 'Computation', ha='center', va='center')

    # Add y-axis bottom text
    # plt.text(0, -2.1, 'Analog', ha='center', va='center')

    # Add y-axis ticks and text
    plt.text(offset_y_types + 0.1, 1.8, 'Develop', ha='center', va='center', size=size_y_types)
    plt.text(offset_y_types + 0.15, -0.15, 'Consume', ha='center', va='center', size=size_y_types)
    plt.text(offset_y_types + 0.15, -1.8, 'Chalk', ha='center', va='center', size=size_y_types)

    # Add small lines intersecting y axis next to text labels
    plt.plot([-0.075, 0], [1.8, 1.8], color='black', lw=2)
    plt.plot([-0.05, 0], [-0.05, 0.0], color='black', lw=2)
    plt.plot([-0.075, 0], [-1.8, -1.8], color='black', lw=2)


    # Add the example numbers and field names
    # plt.text(2.5, -1.8, 'Einstein', ha='center', va='center', size=size_name, color='purple')
    # plt.text(2.4, -1.4, 'Hawking', ha='center', va='center', size=size_name, color='purple')
    # plt.text(2.4, -0.7, 'Ashtekar', ha='center', va='center', size=size_name, color='purple')
    # plt.text(2.3, 0.7, 'Thorne', ha='center', va='center', size=size_name, color='purple')
    # plt.text(1.9, 1.8, 'Pretorius', ha='center', va='center', size=size_name, color='purple')
    # plt.text(1.8, 1.2, 'Sathyaprakash', ha='center', va='center', size=size_name, color='purple')
    plt.text(0, 2.45, '(Examples from LIGO)', ha='center', va='center', size=8, color='purple')
    plt.text(2.1, -1.4, 'General\nRelativity', ha='center', va='center', size=size_name, color='purple')
    plt.text(2.1, 1.4, 'Numerical\nRelativity', ha='center', va='center', size=size_name, color='purple')
    plt.text(-2.1, 1.4, 'Hardware\nInfrastructure', ha='center', va='center', size=size_name, color='purple')
    plt.text(-2.1, -1.4, 'Laser\nConstruction', ha='center', va='center', size=size_name, color='purple')
    plt.text(0, 1.0, 'Data\nAnalysis', ha='center', va='center', size=size_name, color='purple')

    # Add dotted vertical line at x=-1.0
    plt.plot([-1.0, -1.0], [-2, 2], 'k:', lw=1)
    plt.plot([1.0, 1.0], [-2, 2], 'k:', lw=1)


    # Add title
    plt.title('Types of Physicists', pad=40)

    # Add padding to the plot
    plt.tight_layout(pad=1)

    # Show plot
    plt.savefig('types_of_physicists.png', dpi=300)
    plt.show()


def plot_rigor():
    """The below function makes a plot of the different types of physicists. Specific features of the plot include
    the following:
    """
    # Configs
    size_name = 12
    size_y_types = 10
    offset_y_types = -0.5

    # Set the style to XKCD
    plt.xkcd()

    # Hide plot borders
    plt.axis('off')

    # Show x and y axes
    plt.axhline(0, color='black', lw=2)
    plt.axvline(0, color='black', lw=2)

    # Center the plot
    plt.xlim(-0.5, 10)
    plt.ylim(-0.1, 1.1)

    # Compute first line: cumulative normal distribution centered at 5
    x = numpy.arange(0, 10, 0.1)
    y1 = numpy.exp(-0.5 * (x - 5)**2 / 2) + 0.01
    y1 = (y1 / numpy.sum(y1))
    y1 = (0.5 / numpy.max(y1)) * y1


    _y1 = numpy.exp(-0.5 * (x - 6)**2 / 0.5) + 0.01
    _y1 = (_y1 / numpy.sum(_y1))
    _y1 = numpy.cumsum(_y1)
    _y1 = (0.2 / numpy.max(_y1)) * _y1

    y1 = y1 + _y1
    y1 = (0.5 / numpy.max(y1)) * y1 + 0.015
    y2 = numpy.where(x < 5.2, y1, 1 - y1 + 0.03) + 0.025

    # Plot the curves
    plt.plot(x, y2, color='blue', label='Capacity')
    plt.plot(x, y1, color='red', label='Usage')

    # Display plot legend in top left corner and shift to right a bit
    plt.legend(loc='upper right', fontsize=10)#, bbox_to_anchor=(1.0, 0.6))

    # Add vertical dotted line at x = 10/3, 20/3
    plt.plot([3.3, 3.3], [-0.1, 1.1], 'k:', lw=1)
    plt.plot([6.65, 6.65], [-0.1, 1.1], 'k:', lw=1)

    # Plot solid horizontal range lines for periods of education
    h1 = 1.0
    f1 = 0.02
    x11 = 1.5
    x12 = 4.0
    c1 = 'black'
    plt.plot([x11, x12], [h1, h1], '-', lw=2.5, color=c1)
    plt.plot([x11, x11], [h1 - f1, h1 + f1], '-', lw=2.5, color=c1)
    plt.plot([x12, x12], [h1 - f1, h1 + f1], '-', lw=2.5, color=c1)
    plt.text((x11 + x12) / 2, h1, 'Undergrad', ha='center', va='center', size=12, color=c1)

    h2 = 1.0
    f2 = 0.02
    x21 = 4.1
    x22 = 7.5
    c2 = 'black'
    plt.plot([x21, x22], [h2, h2], '-', lw=2.5, color=c2)
    plt.plot([x21, x21], [h2 - f2, h2 + f2], '-', lw=2.5, color=c2)
    plt.plot([x22, x22], [h2 - f2, h2 + f2], '-', lw=2.5, color=c2)
    plt.text((x21 + x22) / 2, h2, 'Graduate', ha='center', va='center', size=12, color=c2)

    # Sub x-axis labels for each stage of rigor
    plt.text(3.3 / 2, -0.1, 'I\nPre - Rigorous', ha='center', va='center', size=12)
    plt.text((3.3 + 6.6) / 2, -0.1, 'II\nRigorous', ha='center', va='center', size=12)
    plt.text((6.6 + 10) /2, -0.1, 'III\nPost - Rigorous', ha='center', va='center', size=12)

    # Add title
    plt.title('Stages of Rigor')

    # Add padding to the plot
    plt.tight_layout(pad=1)

    # Show plot
    plt.savefig('stages_of_rigor.png', dpi=300)
    plt.show()



def main():
    """Main function"""
    plot_broad_types()
    plot_theorist_types()
    plot_rigor()


if __name__ == '__main__':
    main()
