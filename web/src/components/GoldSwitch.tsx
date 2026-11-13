

interface GoldSwitchProps {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  hint?: string
  disabled?: boolean
}

export function GoldSwitch({ checked, onChange, label, hint, disabled }: GoldSwitchProps) {
  return (
    <div className="cf-switch-group">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        className={`cf-switch ${checked ? 'on' : ''} ${disabled ? 'disabled' : ''}`}
        onClick={() => !disabled && onChange(!checked)}
        disabled={disabled}
      >
        <span className="cf-switch-knob" />
      </button>
      <div className="cf-switch-label">
        <span className="cf-switch-title">{label}</span>
        {hint && <span className="cf-switch-hint">{hint}</span>}
      </div>
    </div>
  )
}