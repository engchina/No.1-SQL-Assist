"""Shared Gradio callback helpers."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

import gradio as gr

from utils.vpd_util import require_admin


def _resolved_type_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        return get_type_hints(fn)
    except Exception:
        return {}


def _is_request_annotation(annotation: Any) -> bool:
    if annotation is gr.Request:
        return True

    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return bool(args) and _is_request_annotation(args[0])

    if origin in (Union, UnionType):
        return any(
            arg is not type(None) and _is_request_annotation(arg)
            for arg in get_args(annotation)
        )

    return False


def _request_parameter(
    signature: inspect.Signature,
    hints: dict[str, Any],
) -> str | None:
    for param in signature.parameters.values():
        annotation = hints.get(param.name, param.annotation)
        if _is_request_annotation(annotation):
            return param.name
    return None


def _unique_request_name(signature: inspect.Signature) -> str:
    name = "_admin_request"
    while name in signature.parameters:
        name = f"_{name}"
    return name


def _safe_annotations(
    signature: inspect.Signature,
    hints: dict[str, Any],
    injected_request_name: str | None,
) -> dict[str, Any]:
    annotations: dict[str, Any] = {}
    if injected_request_name is not None:
        annotations[injected_request_name] = gr.Request

    for name, param in signature.parameters.items():
        annotation = hints.get(name, param.annotation)
        if annotation is not inspect.Signature.empty and not isinstance(annotation, str):
            annotations[name] = annotation

    return_annotation = hints.get("return", signature.return_annotation)
    if (
        return_annotation is not inspect.Signature.empty
        and not isinstance(return_annotation, str)
    ):
        annotations["return"] = return_annotation

    return annotations


def _exposed_signature(
    signature: inspect.Signature,
    injected_request_name: str | None,
) -> inspect.Signature:
    if injected_request_name is None:
        return signature

    request_param = inspect.Parameter(
        injected_request_name,
        inspect.Parameter.POSITIONAL_ONLY,
        annotation=gr.Request,
    )
    return signature.replace(
        parameters=[request_param, *signature.parameters.values()]
    )


def _request_index(
    signature: inspect.Signature,
    request_param_name: str | None,
) -> int | None:
    if request_param_name is None:
        return None

    positional_params = [
        param
        for param in signature.parameters.values()
        if param.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    for index, param in enumerate(positional_params):
        if param.name == request_param_name:
            return index
    return None


def _prepare_call(
    signature: inspect.Signature,
    request_param_name: str | None,
    request_param_index: int | None,
    injected_request_name: str | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[Any, tuple[Any, ...], dict[str, Any]]:
    if injected_request_name is not None:
        if args:
            return args[0], args[1:], kwargs
        if injected_request_name in kwargs:
            call_kwargs = dict(kwargs)
            return call_kwargs.pop(injected_request_name), (), call_kwargs
        return None, args, kwargs

    if request_param_name in kwargs:
        return kwargs[request_param_name], args, kwargs
    if request_param_index is not None and len(args) > request_param_index:
        return args[request_param_index], args, kwargs

    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return None, args, kwargs
    return bound.arguments.get(request_param_name), args, kwargs


def admin_only_event(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Inject an ADMIN authorization check while preserving Gradio special args."""
    signature = inspect.signature(fn)
    hints = _resolved_type_hints(fn)
    request_param_name = _request_parameter(signature, hints)
    injected_request_name = (
        None if request_param_name else _unique_request_name(signature)
    )
    request_param_index = _request_index(signature, request_param_name)

    exposed_signature = _exposed_signature(signature, injected_request_name)
    exposed_annotations = _safe_annotations(signature, hints, injected_request_name)

    def authorize(args: tuple[Any, ...], kwargs: dict[str, Any]):
        request, call_args, call_kwargs = _prepare_call(
            signature,
            request_param_name,
            request_param_index,
            injected_request_name,
            args,
            kwargs,
        )
        require_admin(request)
        return call_args, call_kwargs

    if inspect.isasyncgenfunction(fn):

        async def guarded(*args, **kwargs):
            call_args, call_kwargs = authorize(args, kwargs)
            async for item in fn(*call_args, **call_kwargs):
                yield item

    elif inspect.iscoroutinefunction(fn):

        async def guarded(*args, **kwargs):
            call_args, call_kwargs = authorize(args, kwargs)
            return await fn(*call_args, **call_kwargs)

    elif inspect.isgeneratorfunction(fn):

        def guarded(*args, **kwargs):
            call_args, call_kwargs = authorize(args, kwargs)
            yield from fn(*call_args, **call_kwargs)

    else:

        def guarded(*args, **kwargs):
            call_args, call_kwargs = authorize(args, kwargs)
            return fn(*call_args, **call_kwargs)

    functools.update_wrapper(guarded, fn)
    guarded.__signature__ = exposed_signature
    guarded.__annotations__ = exposed_annotations
    return guarded
