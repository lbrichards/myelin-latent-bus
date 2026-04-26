"""Smoke test: the package imports cleanly without a model checkpoint on disk."""


def test_top_level_imports():
    import latent_bus  # noqa: F401
    from latent_bus import (
        ActivationCapture,
        answer_start_injection,
        load_local_model_and_tokenizer,
        select_device_and_dtype,
        set_offline_mode,
    )
    assert callable(answer_start_injection)
    assert callable(load_local_model_and_tokenizer)
    assert callable(select_device_and_dtype)
    assert callable(set_offline_mode)
    assert ActivationCapture is not None


def test_submodule_imports():
    from latent_bus import prepare, run, cli  # noqa: F401
    assert callable(prepare.prepare_fact_vector)
    assert callable(run.run_demo)
    assert callable(run.run_kinship_baseline)
    assert callable(run.run_aligned_patch_control)
    assert callable(run.run_logit_probe)
    assert callable(run.cache_patched_payloads)
    assert callable(run.train_single_payload)
    assert callable(run.train_payload_suite)
    assert callable(run.probe_payload_suite_facts)
    assert callable(cli.prepare_main)
    assert callable(cli.baseline_main)
    assert callable(cli.aligned_patch_main)
    assert callable(cli.cache_payloads_main)
    assert callable(cli.demo_main)
    assert callable(cli.probe_main)
    assert callable(cli.train_payload_main)
    assert callable(cli.train_suite_main)
    assert callable(cli.fact_probe_main)


def test_version():
    import latent_bus
    assert latent_bus.__version__
