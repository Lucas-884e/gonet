package util

func ListConvert[S, T any](s []S) []T {
	t := make([]T, len(s))
	for i, e := range s {
		t[i] = any(e).(T)
	}
	return t
}

func Must1[T any](x T, err error) T {
	if err != nil {
		panic(err)
	}
	return x
}
