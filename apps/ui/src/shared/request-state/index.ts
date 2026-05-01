export type AsyncState<T> =
    | {
        status: "idle";
        data: T | null;
        error: null;
    }
    | {
        status: "loading";
        data: T | null;
        error: null;
    }
    | {
        status: "success";
        data: T;
        error: null;
    }
    | {
        status: "error";
        data: T | null;
        error: string;
    };

export function createIdleState<T>(data: T | null = null): AsyncState<T> {
    return {
        status: "idle",
        data,
        error: null,
    };
}

export function createLoadingState<T>(data: T | null = null): AsyncState<T> {
    return {
        status: "loading",
        data,
        error: null,
    };
}

export function createSuccessState<T>(data: T): AsyncState<T> {
    return {
        status: "success",
        data,
        error: null,
    };
}

export function createErrorState<T>(error: string, data: T | null = null): AsyncState<T> {
    return {
        status: "error",
        data,
        error,
    };
}
