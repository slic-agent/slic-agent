# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import re
from typing import List

# Allowed non-ASCII code points: common technical/math symbols in circuit text.
# These are treated as valid; any other non-ASCII (CJK, Cyrillic, Arabic, etc.) -> invalid.
_ALLOWED_UNICODE_CODE_POINTS = frozenset([
    0x00B0,  # ° degree
    0x00B1,  # ± plus-minus
    0x00B7,  # · middle dot
    0x00D7,  # × multiplication
    0x0394,  # Δ Delta
    0x03A9,  # Ω Ohm (capital omega)
    0x03B1,  # α alpha
    0x03B2,  # β beta
    0x03B5,  # ε epsilon
    0x03B8,  # θ theta
    0x03BB,  # λ lambda
    0x03BC,  # μ micro
    0x03C0,  # π pi
    0x03C1,  # ρ rho
    0x03C3,  # σ sigma
    0x03C4,  # τ tau
    0x03C9,  # ω omega (lowercase)
    0x2014,  # — em dash (punctuation)
    0x2022,  # • bullet (list item)
    0x2032,  # ′ prime
    0x2033,  # ″ double prime
    0x2202,  # ∂ partial
    0x2211,  # ∑ summation
    0x2212,  # − minus (not hyphen)
    0x221A,  # √ square root
    0x221E,  # ∞ infinity
    0x2248,  # ≈ approximately equal
    0x2264,  # ≤ less-than-or-equal
    0x2265,  # ≥ greater-than-or-equal
])

# Too much invalid unicode (garbage, CJK-heavy text, appended junk) makes the text invalid.
_INVALID_UNICODE_RATIO_THRESHOLD = 0.02   # > 2% invalid chars → invalid
_INVALID_UNICODE_MAX_COUNT = 25            # > 25 invalid chars → invalid (catches long garbage)


def _count_invalid_unicode(text: str) -> int:
    """Count chars that are non-ASCII and not in the allowlist."""
    if not text:
        return 0
    return sum(1 for c in text if ord(c) > 127 and ord(c) not in _ALLOWED_UNICODE_CODE_POINTS)


def _is_english_text(text: str) -> bool:
    """True unless the text has more than _INVALID_UNICODE_MAX_COUNT invalid characters (non-ASCII and not
    allowlisted) or an invalid fraction above _INVALID_UNICODE_RATIO_THRESHOLD."""
    if not text:
        return True  # Empty text is considered valid
    n_invalid = _count_invalid_unicode(text)
    if n_invalid == 0:
        return True
    n = len(text)
    if n_invalid > _INVALID_UNICODE_MAX_COUNT:
        return False  # too many invalid chars (e.g. long CJK/garbage)
    if n > 0 and (n_invalid / n) > _INVALID_UNICODE_RATIO_THRESHOLD:
        return False  # too high a fraction (e.g. "correct tag but inside lots of unicode")
    return True


def circuitop_projection(
    actions: List[str],
    action_pools: List[List[str]],
    enforce_think_prefix: bool = False,
    reject_trailing_content: bool = False,
    require_think_tags: bool = True,
    think_prefilled: bool = False,
):
    """
    Parse CircuitOp actions from LLM output: the percentage DSL
       <action>increase @22 w 15%</action>, <action>decrease X0.M3 l 10%</action>
    or <action>success</action> (is_stop_action). An assignment (<action>w_MM30=420n; iref=10u</action>) is
    applied with env.circuitop.action_change_mode=absolute and rejected by the validator otherwise. Returns dict with 'actions': [dict, ...].
    """
    def _response_has_success_action(text: str) -> bool:
        """True if response contains <action>success</action> (the episode's stop action, honoured even when the response is otherwise invalid)."""
        return bool(re.search(r'<action>\s*success\s*</action>', text, re.IGNORECASE))

    valids = [0] * len(actions)
    
    def _extract_proposed_action_raw(text: str) -> str:
        """Best-effort extraction of raw <action>...</action> content for history display."""
        matches = re.findall(r'<action>(.*?)</action>', text, re.DOTALL | re.IGNORECASE)
        if matches:
            return "; ".join(m.strip() for m in matches if m.strip())
        return ""

    assignment_re = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*([^\s;]+)\s*$')

    for i in range(len(actions)):
        original_str = actions[i]
        # think_prefilled: the chat template already ends the prompt with "<think>", so restore the tag
        # and apply every format gate unchanged.
        if think_prefilled and not original_str.lstrip().lower().startswith('<think>'):
            original_str = '<think>' + original_str
        intended_stop = _response_has_success_action(original_str)
        
        try:
            # Optional strict prefix enforcement: first non-whitespace token must be <think>
            if enforce_think_prefix:
                stripped = original_str.lstrip()
                if not stripped.lower().startswith("<think>"):
                    valids[i] = 0
                    out = {
                        'error': 'Response must begin with <think>...</think> reasoning block before any other text.',
                        'proposed_action_raw': _extract_proposed_action_raw(original_str),
                    }
                    if intended_stop:
                        out['is_stop_action'] = True
                    actions[i] = out
                    continue

            # Check for required tags. require_think_tags=False is for reasoner APIs whose reasoning
            # arrives in a separate field, not in the text.
            if require_think_tags and (
                    '<think>' not in original_str.lower() or '</think>' not in original_str.lower()):
                valids[i] = 0
                out = {
                    'error': 'Missing <think></think> tags in response. Please include your reasoning in <think></think> tags before providing an action.',
                    'proposed_action_raw': _extract_proposed_action_raw(original_str),
                }
                if intended_stop:
                    out['is_stop_action'] = True
                actions[i] = out
                continue
            
            # Extract and validate <think> content for English-only (no non-ASCII / unicode)
            think_matches = re.findall(r'<think>(.*?)</think>', original_str, re.DOTALL | re.IGNORECASE)
            if think_matches:
                think_content = think_matches[0].strip()
                if not _is_english_text(think_content):
                    valids[i] = 0
                    out = {
                        'error': 'The content in <think></think> tags must be in English only. Please provide your reasoning in English.',
                        'error_type': 'english_only',
                        'proposed_action_raw': _extract_proposed_action_raw(original_str),
                    }
                    if intended_stop:
                        out['is_stop_action'] = True
                    actions[i] = out
                    continue
            
            # The only accepted layout is <think>…</think> followed by the action block. Reject <action>
            # inside think, </think> after an <action> and duplicated actions (each would execute
            # actions twice).
            _lo = original_str.lower()
            _last_close = _lo.rfind('</think>')
            _first_action = _lo.find('<action>')
            _layout_error = None
            if require_think_tags and _first_action != -1 and _last_close > _first_action:
                _layout_error = ('Malformed response: </think> appears after an <action> block. '
                                 'Write exactly one <think>…</think> block followed by the action block; '
                                 'do not repeat the actions.')
            elif require_think_tags and any('<action>' in t.lower()
                                            for t in re.findall(r'<think>(.*?)</think>', original_str, re.DOTALL | re.IGNORECASE)):
                _layout_error = ('Malformed response: <action> tags inside <think>. Put reasoning only inside '
                                 '<think>…</think> and the action block after it.')
            if _layout_error is not None:
                valids[i] = 0
                out = {'error': _layout_error, 'error_type': 'malformed_layout',
                       'proposed_action_raw': _extract_proposed_action_raw(original_str)}
                if intended_stop:
                    out['is_stop_action'] = True
                actions[i] = out
                continue

            # Extract all <action> blocks (multiple actions per turn allowed)
            action_matches = re.findall(r'<action>(.*?)</action>', original_str, re.DOTALL | re.IGNORECASE)
            _stripped = [m.strip() for m in action_matches]
            if len(_stripped) != len(set(_stripped)):
                valids[i] = 0
                out = {'error': 'Malformed response: the same action is given more than once. '
                                'List each action exactly once.',
                       'error_type': 'duplicate_action',
                       'proposed_action_raw': _extract_proposed_action_raw(original_str)}
                if intended_stop:
                    out['is_stop_action'] = True
                actions[i] = out
                continue
            
            if not action_matches:
                valids[i] = 0
                out = {
                    'error': 'Missing <action></action> block in response',
                    'proposed_action_raw': '',
                }
                if intended_stop:
                    out['is_stop_action'] = True
                actions[i] = out
                continue
            
            # Strip each action content for parsing
            action_contents = [m.strip() for m in action_matches]

            # Optional: reject any non-whitespace content after the last </action> tag
            if reject_trailing_content:
                last_end = None
                for m in re.finditer(r'</action>', original_str, re.IGNORECASE):
                    last_end = m.end()
                if last_end is not None:
                    tail = original_str[last_end:].strip()
                    if tail:
                        valids[i] = 0
                        out = {
                            'error': 'Your response must end immediately after the final </action> tag with no extra text or symbols.',
                            'proposed_action_raw': _extract_proposed_action_raw(original_str),
                        }
                        if intended_stop:
                            out['is_stop_action'] = True
                        actions[i] = out
                        continue
            
            # a single "success" action is the stop action
            if len(action_contents) == 1 and action_contents[0].lower() == 'success':
                actions[i] = {
                    'is_stop_action': True
                }
                valids[i] = 1
                continue
            
            # Parse each action as DSL; collect into list (allow multiple actions per turn)
            parsed_list = []
            parse_error = None
            for action_content in action_contents:
                if action_content.lower() == 'success':
                    parse_error = 'Cannot mix "success" with other actions in the same turn.'
                    break

                # Absolute assignment mode: "param=value; param=value"
                if "=" in action_content:
                    stmts = [s.strip() for s in action_content.split(';') if s.strip()]
                    if not stmts:
                        parse_error = (
                            f'Invalid assignment action "{action_content}". '
                            'Expected at least one assignment: <param>=<value>'
                        )
                        break
                    for stmt in stmts:
                        m = assignment_re.match(stmt)
                        if not m:
                            parse_error = (
                                f'Invalid assignment "{stmt}" in action "{action_content}". '
                                'Expected syntax: <param>=<value> (for example w_MM30=420n)'
                            )
                            break
                        param_raw, value_raw = m.group(1).strip(), m.group(2).strip()
                        parsed_list.append({
                            'param': param_raw,
                            'value': value_raw,
                        })
                    if parse_error:
                        break
                    continue

                parts = action_content.split()
                if len(parts) < 4:
                    missing_fields = ['<direction>', '<target>', '<param>', '<percentage>'][len(parts):]
                    parse_error = f'Missing fields in action: {", ".join(missing_fields)}. Got: "{action_content}". Expected: <direction> <target> <param> <percentage>%'
                    break
                if len(parts) > 4:
                    action_keywords = ['increase', 'decrease']
                    keyword_count = sum(1 for part in parts if part.lower() in action_keywords)
                    if keyword_count > 1:
                        parse_error = f'Multiple actions in one <action> block: "{action_content}". Use one action per <action> </action> tag.'
                    else:
                        parse_error = f'Extra fields in action. Got {len(parts)} fields: "{action_content}". Expected 4: <direction> <target> <param> <percentage>%'
                    break
                direction, target_id, param, pct_str = parts
                param_raw = param.strip()
                if not param_raw:
                    parse_error = (
                        f'Invalid <param_name> "{param}" in action "{action_content}". '
                        f'Use a parameter name token from ACTION_SPACE (e.g., w or Mpair_w)'
                    )
                    break
                if direction.lower() not in ['increase', 'decrease']:
                    parse_error = f'Invalid <direction> "{direction}" in action "{action_content}". Must be "increase" or "decrease"'
                    break
                try:
                    percentage = float(pct_str.rstrip('%'))
                except ValueError:
                    parse_error = f'Invalid <percentage> "{pct_str}" in action "{action_content}". Must be a number followed by % (e.g., "15%")'
                    break
                if not math.isfinite(percentage):
                    parse_error = (
                        f'Invalid <percentage> "{pct_str}" in action "{action_content}". '
                        'Percentage must be a finite number.'
                    )
                    break
                parsed_list.append({
                    'direction': direction.lower(),
                    'target_id': target_id,
                    'param': param_raw,
                    'percentage': percentage,
                })
            
            if parse_error:
                valids[i] = 0
                out = {
                    'error': parse_error,
                    'proposed_action_raw': "; ".join(action_contents),
                }
                if intended_stop:
                    out['is_stop_action'] = True
                actions[i] = out
                continue
            
            # Return list of actions; env will apply them in sequence
            actions[i] = {
                'actions': parsed_list,
                'is_stop_action': False
            }
            valids[i] = 1
            
        except Exception as e:
            valids[i] = 0
            out = {
                'error': f'Parsing error: {str(e)}',
                'proposed_action_raw': _extract_proposed_action_raw(original_str),
            }
            if intended_stop:
                out['is_stop_action'] = True
            actions[i] = out
    
    return actions, valids
