package an

import (
	"fmt"
	"testing"
)

func TestStability(t *testing.T) {
	Z := [][]float64{{1, 1, 0, 0, 1, 1, 0, 0, 1}, {1, 1, 1, 0, 0, 1, 1, 0, 0}, {0, 1, 1, 1, 0, 1, 1, 1, 0}, {0, 1, 1, 0, 0, 1, 1, 0, 1}, {1, 0, 1, 1, 0, 0, 0, 0, 1}}
	fmt.Println(StabilityAndVariance(Z))
	fmt.Println("Should be: stability': 0.009999999999999898, variance: 0.014128020000000002")
}

func TestGrouping(t *testing.T) {
	group := [][]int{{1, 2}, {3, 4}, {5, 6}}
	input := []int{1, 2, 5, 6}

	f := MakeGrouperFunc(group)

	fmt.Println(f(input))
	fmt.Println("Should be []int{0,2},3")

}
