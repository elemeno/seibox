"""Tests for profile functionality."""

import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from seibox.utils.schemas import ProfileConfig, InputRecord
from seibox.runners.eval_runner import apply_mitigations_pre, apply_mitigations_post


def test_profile_config():
    """Test ProfileConfig creation and defaults."""
    # Test with defaults
    profile = ProfileConfig()
    assert profile.prompt_hardening is False
    assert profile.policy_gate is False
    
    # Test with explicit values
    profile = ProfileConfig(prompt_hardening=True, policy_gate=True)
    assert profile.prompt_hardening is True
    assert profile.policy_gate is True


def test_profiles_yaml_loading():
    """Test loading profiles from profiles.yaml."""
    profiles_path = Path("configs/profiles.yaml")
    assert profiles_path.exists(), "profiles.yaml should exist"
    
    with open(profiles_path, "r") as f:
        profiles_config = yaml.safe_load(f)
    
    # Check all expected profiles exist
    expected_profiles = ["baseline", "policy_gate", "prompt_hardening", "both"]
    assert "profiles" in profiles_config
    for profile_name in expected_profiles:
        assert profile_name in profiles_config["profiles"]
    
    # Validate each profile can be loaded as ProfileConfig
    for name, data in profiles_config["profiles"].items():
        profile = ProfileConfig(**data)
        
        # Validate specific profiles
        if name == "baseline":
            assert profile.prompt_hardening is False
            assert profile.policy_gate is False
        elif name == "both":
            assert profile.prompt_hardening is True
            assert profile.policy_gate is True


@patch('seibox.runners.eval_runner.policy_gate')
@patch('seibox.runners.eval_runner.prompt_hardening')
def test_apply_mitigations_with_profile(mock_prompt_hardening, mock_policy_gate):
    """Test that profiles correctly apply mitigations."""
    # Setup mocks
    mock_policy_gate.apply_pre.return_value = ("modified_prompt", {"gate_applied": True})
    mock_policy_gate.apply_post.return_value = ("modified_text", {"post_gate_applied": True})
    mock_prompt_hardening.get_system_prompt.return_value = "Safety system prompt"
    
    # Test with baseline profile (no mitigations)
    profile = ProfileConfig(prompt_hardening=False, policy_gate=False)
    prompt, system_prompt, trace = apply_mitigations_pre("test prompt", None, {}, profile)
    
    assert prompt == "test prompt"  # Unchanged
    assert system_prompt is None
    assert trace["mitigations"] == []
    mock_policy_gate.apply_pre.assert_not_called()
    mock_prompt_hardening.get_system_prompt.assert_not_called()
    
    # Reset mocks
    mock_policy_gate.reset_mock()
    mock_prompt_hardening.reset_mock()
    
    # Test with both profile (all mitigations)
    profile = ProfileConfig(prompt_hardening=True, policy_gate=True)
    prompt, system_prompt, trace = apply_mitigations_pre("test prompt", None, {}, profile)
    
    assert prompt == "modified_prompt"
    assert system_prompt == "Safety system prompt"
    assert "policy_gate@0.1.0:pre" in trace["mitigations"]
    assert "prompt_hardening@0.1.0" in trace["mitigations"]
    mock_policy_gate.apply_pre.assert_called_once()
    mock_prompt_hardening.get_system_prompt.assert_called_once_with("default_v0")
    
    # Test post-processing with profile
    text, post_trace = apply_mitigations_post("response text", None, "test prompt", profile)
    assert text == "modified_text"
    assert "policy_gate@0.1.0:post" in post_trace["mitigations"]
    mock_policy_gate.apply_post.assert_called_once()


def test_profile_precedence():
    """Test that profile takes precedence over mitigation_id."""
    with patch('seibox.runners.eval_runner.policy_gate') as mock_policy_gate:
        with patch('seibox.runners.eval_runner.prompt_hardening') as mock_prompt_hardening:
            mock_policy_gate.apply_pre.return_value = ("modified", {})
            mock_prompt_hardening.get_system_prompt.return_value = "system"
            
            # When both profile and mitigation_id are provided, profile should be used
            profile = ProfileConfig(prompt_hardening=False, policy_gate=True)
            prompt, system_prompt, trace = apply_mitigations_pre(
                "test", 
                "prompt_hardening@0.1.0",  # This should be ignored
                {}, 
                profile
            )
            
            # Profile says no prompt hardening, so system_prompt should be None
            assert system_prompt is None
            # Profile says policy gate, so it should be applied
            assert prompt == "modified"
            mock_policy_gate.apply_pre.assert_called_once()
            mock_prompt_hardening.get_system_prompt.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])